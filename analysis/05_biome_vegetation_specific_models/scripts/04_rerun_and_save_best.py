"""Rerun and save best vegetation-specific models from Optuna studies.

For a given trait, this script loads each (group, fold) study, extracts the
best completed trial, rebuilds the training configuration from that trial,
refits the sklearn pipeline, and saves the final model as pickle for deployment
and downstream evaluation.

Run from project root:
    python analysis/05_biome_vegetation_specific_models/scripts/04_rerun_and_save_best.py --trait laie
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import optuna
import pandas as pd
import yaml
from loguru import logger
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from sklearn.model_selection import GroupKFold, train_test_split

BASE_DIR_ALL = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR_ALL))

BASE_DIR_ANALYSIS = Path(__file__).resolve().parents[1]
VEG_CONFIG_PATH = BASE_DIR_ANALYSIS / "config" / "train_pipeline_veg.yaml"
CV_SPLITS_PATH = BASE_DIR_ANALYSIS / "data" / "cv_splits.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR_ANALYSIS / "results" / "models"

from train_pipeline.optuna_training import (
    initialize_pipeline,
    optimize_hyperparams,
    predict_and_evaluate,
    prepare_features_and_target,
    simulate_data,
)
from train_pipeline.utils_loading import load_grounded_eo_validation_data


def add_global_group9_splits(
    cv_splits: pd.DataFrame,
    df_val: pd.DataFrame,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Add synthetic group 9 with ecoregion-based folds across groups 1..8 samples."""
    cv = cv_splits.copy()
    cv["RECODED_GROUP"] = cv["RECODED_GROUP"].astype(int)
    cv = cv[cv["test_fold"] >= 0]
    cv = cv[cv["RECODED_GROUP"] != 9]

    base = cv[cv["RECODED_GROUP"].isin(range(1, 9))][["uuid", "ECO_ID"]].drop_duplicates()
    if base.empty:
        raise RuntimeError("No base samples found in groups 1..8 for creating group 9 splits")

    available = df_val[["uuid", "ECO_ID"]].drop_duplicates()
    base = base.merge(available, on=["uuid", "ECO_ID"], how="inner")
    if base.empty:
        raise RuntimeError("No overlap between cv_splits groups 1..8 and validation data")

    gkf = GroupKFold(n_splits=n_splits)
    fold_by_uuid: dict[str, int] = {}
    X_dummy = base[["uuid"]]
    y_dummy = pd.Series([0] * len(base))

    for fold_id, (_, test_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=base["ECO_ID"])):
        uuids_fold = base.iloc[test_idx]["uuid"].tolist()
        for u in uuids_fold:
            fold_by_uuid[u] = fold_id

    group9 = base.copy()
    group9["RECODED_GROUP"] = 9
    group9["test_fold"] = group9["uuid"].map(fold_by_uuid)

    return pd.concat(
        [cv, group9[["uuid", "ECO_ID", "RECODED_GROUP", "test_fold"]]],
        ignore_index=True,
    )


def load_veg_config() -> dict:
    with open(VEG_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_study_name(config: dict, group_id: int, fold_id: int) -> str:
    return (
        config["optuna_study_name"]
        .replace("(TRAIT)", config["trait"])
        .replace("(GROUP)", str(group_id))
        .replace("(FOLD)", str(fold_id))
        .replace("(MODEL)", config["model"])
    )


def prepare_validation_data_for_group(
    df_val: pd.DataFrame,
    cv_splits: pd.DataFrame,
    group_id: int,
    fold_id: int,
    trait: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    splits_group = cv_splits[cv_splits["RECODED_GROUP"] == group_id]
    if splits_group.empty:
        raise ValueError(f"No cv_splits rows found for group_id={group_id}")

    test_uuids = set(splits_group[splits_group["test_fold"] == fold_id]["uuid"])

    df_group = df_val[df_val["uuid"].isin(splits_group["uuid"])]
    df_val_test = df_group[df_group["uuid"].isin(test_uuids)]
    df_val_train = df_group[~df_group["uuid"].isin(test_uuids)]

    return (
        df_val_train[bands + angles],
        df_val_test[bands + angles],
        df_val_train[[trait]],
        df_val_test[[trait]],
    )


def parse_args() -> argparse.Namespace:
    # parser = argparse.ArgumentParser(description=__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trait",
        choices=["fapar", "fcover", "laie"],
        default="laie",
        help="Trait to rerun and save best models for (default: laie).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of RECODED_GROUP IDs to process.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of fold IDs to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model artifacts are saved.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip retraining if target pickle already exists.",
    )
    return parser.parse_args()


def get_best_completed_trial(study: optuna.Study):
    completed = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        return None
    return min(completed, key=lambda t: t.value)


def save_artifacts(
    model,
    config: dict,
    study_name: str,
    output_dir: Path,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"model_{study_name}.pkl"
    config_path = output_dir / f"model_{study_name}_config.json"
    metadata_path = output_dir / f"model_{study_name}_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def rerun_and_save_one(
    base_config: dict,
    storage: RDBStorage,
    df_val: pd.DataFrame,
    cv_splits: pd.DataFrame,
    trait: str,
    group_id: int,
    fold_id: int,
    output_dir: Path,
    skip_existing: bool,
) -> None:
    config_for_name = dict(base_config)
    config_for_name["trait"] = trait
    study_name = resolve_study_name(config_for_name, group_id, fold_id)

    trait_output_dir = output_dir / trait
    model_path = trait_output_dir / f"model_{study_name}.pkl"
    if skip_existing and model_path.exists():
        logger.info(f"Skipping existing model: {model_path}")
        return

    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = get_best_completed_trial(study)
    if best_trial is None:
        logger.warning(f"No completed trials in study '{study_name}'. Skipping.")
        return

    if "config" not in best_trial.user_attrs:
        raise KeyError(
            f"Study '{study_name}' best trial has no 'config' in user_attrs. "
            "Cannot reconstruct exact training configuration."
        )

    best_config = dict(best_trial.user_attrs["config"])
    best_config["trait"] = trait

    sim_trait = "lai" if trait == "laie" else trait
    df_sim = simulate_data(best_config, sim_trait)
    X, y = prepare_features_and_target(df_sim, sim_trait)

    # Keep identical data split logic to the Optuna objective before refitting on all data.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)

    pipeline = initialize_pipeline(best_config)
    best_model = optimize_hyperparams(pipeline, X_train, y_train, best_config)
    best_model.set_params(regressor__regressor__max_iter=10000)
    model_all = best_model.fit(X, y.squeeze())

    X_val_train, X_val_test, y_val_train, y_val_test = (
        prepare_validation_data_for_group(
            df_val=df_val,
            cv_splits=cv_splits,
            group_id=group_id,
            fold_id=fold_id,
            trait=trait,
        )
    )
    _, val_scores = predict_and_evaluate(
        model_all,
        X_val_train,
        X_val_test,
        y_val_train,
        y_val_test,
        trait,
    )

    completed_trials = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    metadata = {
        "study_name": study_name,
        "trait": trait,
        "group_id": group_id,
        "fold_id": fold_id,
        "best_trial_number": best_trial.number,
        "best_trial_value": float(best_trial.value),
        "n_complete_trials": int(completed_trials),
        "optuna_params": best_trial.params,
        "n_samples_simulated": len(X),
        "n_samples_val_train": len(X_val_train),
        "n_samples_val_test": len(X_val_test),
        "val_scores": {k: float(v) for k, v in val_scores.items()},
    }

    save_artifacts(
        model=model_all,
        config=best_config,
        study_name=study_name,
        output_dir=trait_output_dir,
        metadata=metadata,
    )
    logger.info(
        f"Saved model for trait={trait}, group={group_id}, fold={fold_id} | "
        f"best_trial={best_trial.number}, best_value={best_trial.value:.4f}"
    )


def main() -> None:
    args = parse_args()
    base_config = load_veg_config()
    base_config["trait"] = args.trait

    df_val = load_grounded_eo_validation_data().rename(
        columns={"phi": "psi", "sza": "tts", "vza": "tto"}
    )
    cv_splits = pd.read_csv(CV_SPLITS_PATH)
    cv_splits = add_global_group9_splits(cv_splits=cv_splits, df_val=df_val, n_splits=3)

    group_ids = sorted(cv_splits["RECODED_GROUP"].unique().tolist())
    fold_ids = sorted(cv_splits["test_fold"].unique().tolist())
    if args.groups is not None:
        group_ids = sorted(set(args.groups))
    if args.folds is not None:
        fold_ids = sorted(set(args.folds))

    storage = RDBStorage(url=base_config["optuna_storage"])

    logger.info(
        f"Rerun+save best models | trait={args.trait} | "
        f"groups={group_ids} | folds={fold_ids}"
    )
    logger.info(f"Output dir: {args.output_dir / args.trait}")

    failures: list[tuple[int, int, str]] = []
    for group_id in group_ids:
        for fold_id in fold_ids:
            try:
                rerun_and_save_one(
                    base_config=base_config,
                    storage=storage,
                    df_val=df_val,
                    cv_splits=cv_splits,
                    trait=args.trait,
                    group_id=group_id,
                    fold_id=fold_id,
                    output_dir=args.output_dir,
                    skip_existing=args.skip_existing,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    f"Failed rerun for trait={args.trait}, group={group_id}, fold={fold_id}: {exc}"
                )
                failures.append((group_id, fold_id, str(exc)))

    if failures:
        logger.error("Rerun completed with failures:")
        for group_id, fold_id, msg in failures:
            logger.error(f"  group={group_id} fold={fold_id}: {msg}")
        raise SystemExit(1)

    logger.info("Rerun and save completed for all requested studies.")


if __name__ == "__main__":
    main()
