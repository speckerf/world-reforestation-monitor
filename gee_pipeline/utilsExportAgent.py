import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

import ee
from loguru import logger


# --- Small helper record to persist / track per tile
@dataclass
class TileJob:
    mgrs_tile: str
    task_id: Optional[str] = None
    description: Optional[str] = None  # your system_index
    asset_id: Optional[str] = None
    state: str = "NOT_SENT"  # NOT_SENT | READY | RUNNING | COMPLETED | FAILED | CANCELLED | UNKNOWN
    error_message: Optional[str] = None
    last_update_utc: Optional[float] = None  # time.time()
    attempts: int = 0


class MGRSExportAgent:
    """
    Runs MGRS exports with a max number of in-flight tasks (RUNNING + READY).
    Polls task states and submits more when capacity is available.
    Emits a summary every `summary_every_s` seconds.
    Optionally persists state to disk (recommended for long runs).
    """

    def __init__(
        self,
        mgrs_tiles: List[str],
        export_fn,  # function: (mgrs_tile: str) -> ee.batch.Task
        max_inflight: int = 20,
        poll_every_s: int = 60,
        summary_every_s: int = 60,
        persist_path: Optional[str] = "mgrs_export_agent_state.json",
        max_attempts: int = 2,
        backoff_s: int = 10,
    ):
        self.export_fn = export_fn
        self.max_inflight = max_inflight
        self.poll_every_s = poll_every_s
        self.summary_every_s = summary_every_s
        self.persist_path = persist_path
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s

        # mgrs_tile -> TileJob
        self.jobs: Dict[str, TileJob] = {t: TileJob(mgrs_tile=t) for t in mgrs_tiles}

        # If there is a persisted state, load it and merge with current list
        self._load_state()

        # For summary cadence
        self._last_summary = 0.0

    # ----------------- Persistence -----------------
    def _load_state(self):
        if not self.persist_path:
            return
        if not os.path.exists(self.persist_path):
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            loaded = {d["mgrs_tile"]: TileJob(**d) for d in data.get("jobs", [])}
            # Merge: keep current tiles, overlay loaded data where applicable
            for t, job in loaded.items():
                if t in self.jobs:
                    self.jobs[t] = job
            logger.info(f"Loaded agent state from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Could not load state {self.persist_path}: {e}")

    def _save_state(self):
        if not self.persist_path:
            return
        try:
            payload = {"jobs": [asdict(j) for j in self.jobs.values()]}
            tmp = self.persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.persist_path)
        except Exception as e:
            logger.warning(f"Could not save state {self.persist_path}: {e}")

    # ----------------- Task polling -----------------
    @staticmethod
    def _get_all_ee_tasks() -> Dict[str, dict]:
        """
        Returns dict task_id -> status dict for all tasks visible to the account.
        """
        statuses = ee.data.getTaskList()
        return {s.get("id"): s for s in statuses if s.get("id")}

    def _refresh_states(self):
        """
        Update each job state based on EE task list.
        """
        task_map = self._get_all_ee_tasks()
        now = time.time()

        for job in self.jobs.values():
            if not job.task_id:
                # Not submitted yet
                continue

            status = task_map.get(job.task_id)
            if not status:
                # Task not found: could be very old, or the list is partial for some reason.
                # Keep previous state but mark unknown.
                if job.state not in {"COMPLETED", "FAILED", "CANCELLED"}:
                    job.state = "UNKNOWN"
                    job.last_update_utc = now
                continue

            state = status.get("state", "UNKNOWN")
            job.state = state
            job.last_update_utc = now

            err = status.get("error_message") or status.get("errorMessage")
            if err:
                job.error_message = err

        self._save_state()

    # ----------------- Capacity / submission -----------------
    def _count_inflight(self) -> Tuple[int, int]:
        running = sum(1 for j in self.jobs.values() if j.state == "RUNNING")
        ready = sum(1 for j in self.jobs.values() if j.state == "READY")
        return running, ready

    def _eligible_to_submit(self, job: TileJob) -> bool:
        if job.state == "NOT_SENT":
            return True
        # retry failed tasks up to max_attempts
        if job.state in {"FAILED", "CANCELLED"} and job.attempts < self.max_attempts:
            return True
        # If unknown (task missing), you might want to retry as well:
        if job.state == "UNKNOWN" and job.attempts < self.max_attempts:
            return True
        return False

    def _submit_one(self, job: TileJob) -> bool:
        """
        Submit one export; update job with task_id and new state guess.
        Returns True if submitted.
        """
        try:
            task = self.export_fn(job.mgrs_tile)  # must return ee.batch.Task
            if task is None:
                # export_fn may return None when collection empty etc.
                job.state = "FAILED"
                job.error_message = (
                    "export_fn returned None (likely no images after filtering)"
                )
                job.attempts += 1
                job.last_update_utc = time.time()
                self._save_state()
                return False

            # In the Python API, task.id is usually available after start()
            job.task_id = getattr(task, "id", None) or getattr(task, "task_id", None)

            # Your export code uses system_index as `description`
            job.description = (
                getattr(task, "config", {}).get("description")
                if hasattr(task, "config")
                else job.description
            )

            job.attempts += 1
            job.state = "READY"  # optimistic; will be corrected on next poll
            job.error_message = None
            job.last_update_utc = time.time()
            self._save_state()

            logger.info(f"Submitted {job.mgrs_tile} -> task_id={job.task_id}")
            return True

        except Exception as e:
            job.state = "FAILED"
            job.error_message = str(e)
            job.attempts += 1
            job.last_update_utc = time.time()
            self._save_state()
            logger.exception(f"Submission failed for {job.mgrs_tile}: {e}")
            return False

    def _fill_capacity(self):
        """
        Submit as many jobs as needed so that RUNNING+READY <= max_inflight.
        """
        running, ready = self._count_inflight()
        inflight = running + ready
        capacity = self.max_inflight - inflight
        if capacity <= 0:
            return

        # Submit in deterministic order (sorted tile names) for reproducibility
        candidates = [j for j in self.jobs.values() if self._eligible_to_submit(j)]
        candidates.sort(key=lambda j: j.mgrs_tile)

        if not candidates:
            return

        n = min(capacity, len(candidates))
        logger.info(f"Capacity available: {capacity}. Submitting {n} job(s).")

        submitted = 0
        for job in candidates[:n]:
            ok = self._submit_one(job)
            if ok:
                submitted += 1
            time.sleep(self.backoff_s)  # gentle spacing to avoid bursty API behavior

        if submitted:
            # refresh soon after submitting to get accurate READY/RUNNING counts
            self._refresh_states()

    # ----------------- Reporting -----------------
    def _summary_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for j in self.jobs.values():
            counts[j.state] = counts.get(j.state, 0) + 1
        return counts

    def _log_summary(self):
        counts = self._summary_counts()
        running, ready = self._count_inflight()
        total = len(self.jobs)

        # Normalize labels you asked for:
        completed = counts.get("COMPLETED", 0)
        pending = counts.get("READY", 0)  # queued
        not_sent = counts.get("NOT_SENT", 0)
        failed = counts.get("FAILED", 0) + counts.get("CANCELLED", 0)
        unknown = counts.get("UNKNOWN", 0)

        logger.info(
            f"[SUMMARY] total={total} | running={running} | pending(READY)={pending} | "
            f"completed={completed} | not_sent={not_sent} | failed/cancelled={failed} | unknown={unknown}"
        )

    # ----------------- Main loop -----------------
    def run(self):
        """
        Run until all jobs are COMPLETED or exhausted (FAILED/CANCELLED with attempts maxed).
        """
        logger.info(
            f"Starting MGRSExportAgent with max_inflight={self.max_inflight}, "
            f"poll_every_s={self.poll_every_s}, summary_every_s={self.summary_every_s}"
        )

        while True:
            self._refresh_states()
            self._fill_capacity()

            now = time.time()
            if now - self._last_summary >= self.summary_every_s:
                self._log_summary()
                self._last_summary = now

            # Stop condition:
            unfinished = [
                j
                for j in self.jobs.values()
                if j.state not in {"COMPLETED"}
                and not (
                    j.state in {"FAILED", "CANCELLED", "UNKNOWN"}
                    and j.attempts >= self.max_attempts
                )
            ]
            if not unfinished:
                self._log_summary()
                logger.info("All jobs finished (completed or exhausted retries).")
                return

            time.sleep(self.poll_every_s)
