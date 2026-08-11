"""Coordinate parallel vegetation-specific Optuna studies.

This script launches multiple runner processes from:
    analysis/05_biome_vegetation_specific_models/scripts/03_train_models.py

Each runner handles one (trait, group, fold) study. Because every worker gets a
separate study name, parallel runs do not contend for the same study and each
study stops when its configured total complete trials is reached.

Examples:
    python analysis/05_biome_vegetation_specific_models/scripts/04_train_models_coordinator.py --num-workers 6
    python analysis/05_biome_vegetation_specific_models/scripts/04_train_models_coordinator.py --traits fapar laie --num-workers 4
    python analysis/05_biome_vegetation_specific_models/scripts/04_train_models_coordinator.py --groups 1 2 3 --folds 0 1 --num-workers 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER_SCRIPT = Path(__file__).resolve().parent / "03a_train_models.py"

VALID_TRAITS = ("fapar", "fcover", "laie")
DEFAULT_GROUPS = (6, 5, 4, 3, 8, 7, 1, 2, 9)
DEFAULT_FOLDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vegetation-specific model training in parallel workers."
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        choices=VALID_TRAITS,
        default=list(VALID_TRAITS),
        help="Traits to run.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=int,
        default=list(DEFAULT_GROUPS),
        help="Recoded group IDs to run (default includes synthetic global group 9).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(DEFAULT_FOLDS),
        help="Fold IDs to run (default: 0 1 2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Maximum number of parallel runner processes.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop scheduling new tasks after the first failed runner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file path to write a full coordinator log.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path | None) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="INFO",
            enqueue=True,
            rotation="50 MB",
            retention="10 days",
            compression="zip",
        )


def run_task(
    task: tuple[str, int, int],
) -> tuple[tuple[str, int, int], int, float, str, str]:
    trait, group_id, fold_id = task
    cmd = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--trait",
        trait,
        "--group",
        str(group_id),
        "--fold",
        str(fold_id),
    ]

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    duration_sec = time.monotonic() - start
    return task, result.returncode, duration_sec, result.stdout, result.stderr


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    tasks: list[tuple[str, int, int]] = [
        (trait, group_id, fold_id)
        for trait in args.traits
        for group_id in args.groups
        for fold_id in args.folds
    ]

    logger.info(f"Runner script: {RUNNER_SCRIPT}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(
        "Planned tasks: "
        f"traits={args.traits}, groups={args.groups}, folds={args.folds} "
        f"=> {len(tasks)} studies"
    )
    logger.info(f"Max workers: {args.num_workers}")
    if args.log_file is not None:
        logger.info(f"Log file: {args.log_file}")

    if args.dry_run:
        for trait, group_id, fold_id in tasks:
            logger.info(
                f"DRY RUN | {sys.executable} {RUNNER_SCRIPT} "
                f"--trait {trait} --group {group_id} --fold {fold_id}"
            )
        return

    failures: list[tuple[str, int, int, int]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        future_to_task = {pool.submit(run_task, task): task for task in tasks}

        for idx, future in enumerate(
            concurrent.futures.as_completed(future_to_task), start=1
        ):
            task = future_to_task[future]
            trait, group_id, fold_id = task

            try:
                _, returncode, duration_sec, stdout, stderr = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    f"Task {idx}/{len(tasks)} failed with exception "
                    f"for trait={trait} group={group_id} fold={fold_id}: {exc}"
                )
                failures.append((trait, group_id, fold_id, 1))
                if args.stop_on_error:
                    logger.error("Stopping early because --stop-on-error is set.")
                    break
                continue

            if returncode == 0:
                if stdout.strip():
                    logger.info(
                        f"stdout | trait={trait} group={group_id} fold={fold_id}\n{stdout.rstrip()}"
                    )
                if stderr.strip():
                    logger.info(
                        f"stderr | trait={trait} group={group_id} fold={fold_id}\n{stderr.rstrip()}"
                    )
                logger.info(
                    f"Task {idx}/{len(tasks)} done | trait={trait} "
                    f"group={group_id} fold={fold_id} | {duration_sec:.1f}s"
                )
            else:
                if stdout.strip():
                    logger.error(
                        f"stdout | trait={trait} group={group_id} fold={fold_id}\n{stdout.rstrip()}"
                    )
                if stderr.strip():
                    logger.error(
                        f"stderr | trait={trait} group={group_id} fold={fold_id}\n{stderr.rstrip()}"
                    )
                logger.error(
                    f"Task {idx}/{len(tasks)} failed (code {returncode}) | "
                    f"trait={trait} group={group_id} fold={fold_id} | {duration_sec:.1f}s"
                )
                failures.append((trait, group_id, fold_id, returncode))
                if args.stop_on_error:
                    logger.error("Stopping early because --stop-on-error is set.")
                    break

    if failures:
        logger.error("Some studies failed:")
        for trait, group_id, fold_id, code in failures:
            logger.error(
                f"  trait={trait} group={group_id} fold={fold_id} return_code={code}"
            )
        raise SystemExit(1)

    logger.info("All scheduled studies completed successfully.")


if __name__ == "__main__":
    main()
