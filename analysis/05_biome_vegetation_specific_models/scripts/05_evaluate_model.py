"""Evaluate vegetation-specific models on stacked OOS and ensemble predictions.

Workflow for a given trait:
1. Load all saved fold models.
2. For each model, determine calibration vs validation (OOS) samples using
   group+fold definitions from cv_splits.
3. Predict OOS split for each model and stack all OOS predictions.
4. Predict every validation sample with every fold model and compute ensemble
   metrics from the mean prediction across all models.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

BASE_DIR_ALL = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR_ALL))

BASE_DIR_ANALYSIS = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = BASE_DIR_ANALYSIS / "results" / "models"
DEFAULT_RESULTS_DIR = BASE_DIR_ANALYSIS / "results" / "evaluation"
DEFAULT_CV_SPLITS_PATH = BASE_DIR_ANALYSIS / "data" / "cv_splits.csv"

from train_pipeline.utils_loading import load_grounded_eo_validation_data

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
ANGLES = ["tts", "tto", "psi"]
FEATURES = BANDS + ANGLES


@dataclass
class ModelArtifact:
    study_name: str
    group_id: int
    fold_id: int
    path_model: Path
    path_metadata: Path
    path_config: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trait",
        type=str,
        choices=["fapar", "fcover", "laie"],
        default="laie",
        help="Trait to evaluate.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root model directory created by 04_rerun_and_save_best.py.",
    )
    parser.add_argument(
        "--cv-splits",
        type=Path,
        default=DEFAULT_CV_SPLITS_PATH,
        help="Path to cv_splits.csv with uuid/ECO_ID/RECODED_GROUP/test_fold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write evaluation tables.",
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "n": float(len(y_true)),
    }


def load_model_artifacts(models_dir: Path, trait: str) -> list[ModelArtifact]:
    trait_dir = models_dir / trait
    if not trait_dir.exists():
        raise FileNotFoundError(f"Trait model directory not found: {trait_dir}")

    artifacts: list[ModelArtifact] = []
    for path_metadata in sorted(trait_dir.glob("model_*_metadata.json")):
        with open(path_metadata) as f:
            metadata = json.load(f)

        study_name = metadata["study_name"]
        group_id = int(metadata["group_id"])
        fold_id = int(metadata["fold_id"])

        path_model = trait_dir / f"model_{study_name}.pkl"
        path_config = trait_dir / f"model_{study_name}_config.json"

        if not path_model.exists():
            raise FileNotFoundError(
                f"Missing model pickle for study '{study_name}': {path_model}"
            )
        if not path_config.exists():
            raise FileNotFoundError(
                f"Missing model config for study '{study_name}': {path_config}"
            )

        artifacts.append(
            ModelArtifact(
                study_name=study_name,
                group_id=group_id,
                fold_id=fold_id,
                path_model=path_model,
                path_metadata=path_metadata,
                path_config=path_config,
            )
        )

    if not artifacts:
        raise RuntimeError(f"No model metadata files found in {trait_dir}")

    return artifacts


def load_eval_dataframe(cv_splits_path: Path, trait: str) -> pd.DataFrame:
    df_val = load_grounded_eo_validation_data().rename(
        columns={"phi": "psi", "sza": "tts", "vza": "tto"}
    )

    required_val_cols = ["uuid", "ECO_ID", trait] + FEATURES
    missing_val_cols = [c for c in required_val_cols if c not in df_val.columns]
    if missing_val_cols:
        raise KeyError(f"Validation data missing columns: {missing_val_cols}")

    cv_splits = pd.read_csv(cv_splits_path)
    cv_splits["RECODED_GROUP"] = cv_splits["RECODED_GROUP"].astype(int)
    cv_splits = cv_splits[cv_splits["test_fold"] >= 0]

    required_cv_cols = ["uuid", "ECO_ID", "RECODED_GROUP", "test_fold"]
    missing_cv_cols = [c for c in required_cv_cols if c not in cv_splits.columns]
    if missing_cv_cols:
        raise KeyError(f"cv_splits missing columns: {missing_cv_cols}")

    df_eval = df_val.merge(
        cv_splits[required_cv_cols],
        on=["uuid", "ECO_ID"],
        how="inner",
    )

    if df_eval.empty:
        raise RuntimeError("No overlapping rows between validation data and cv_splits")

    return df_eval


def evaluate_models(
    df_eval: pd.DataFrame,
    artifacts: list[ModelArtifact],
    trait: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_all = df_eval[FEATURES]
    y_all = df_eval[trait].to_numpy()

    per_model_rows: list[dict] = []
    split_rows: list[dict] = []

    stacked_y_true: list[np.ndarray] = []
    stacked_y_pred: list[np.ndarray] = []
    stacked_uuid: list[str] = []
    stacked_study: list[str] = []

    ensemble_preds: list[np.ndarray] = []

    for artifact in artifacts:
        with open(artifact.path_model, "rb") as f:
            model = pickle.load(f)

        y_pred_all = np.asarray(model.predict(X_all)).squeeze()
        ensemble_preds.append(y_pred_all)

        mask_group = df_eval["RECODED_GROUP"] == artifact.group_id
        mask_oos = mask_group & (df_eval["test_fold"] == artifact.fold_id)
        mask_cal = mask_group & (df_eval["test_fold"] != artifact.fold_id)

        y_true_oos = df_eval.loc[mask_oos, trait].to_numpy()
        y_pred_oos = y_pred_all[mask_oos.to_numpy()]

        y_true_cal = df_eval.loc[mask_cal, trait].to_numpy()
        y_pred_cal = y_pred_all[mask_cal.to_numpy()]

        if len(y_true_oos) == 0:
            logger.warning(
                f"No OOS samples for study={artifact.study_name} "
                f"group={artifact.group_id} fold={artifact.fold_id}. Skipping OOS row."
            )
        else:
            row_oos = {
                "trait": trait,
                "study_name": artifact.study_name,
                "group_id": artifact.group_id,
                "fold_id": artifact.fold_id,
                "split": "validation_oos",
                **compute_metrics(y_true_oos, y_pred_oos),
            }
            per_model_rows.append(row_oos)

            stacked_y_true.append(y_true_oos)
            stacked_y_pred.append(y_pred_oos)
            stacked_uuid.extend(df_eval.loc[mask_oos, "uuid"].tolist())
            stacked_study.extend([artifact.study_name] * len(y_true_oos))

        if len(y_true_cal) > 0:
            row_cal = {
                "trait": trait,
                "study_name": artifact.study_name,
                "group_id": artifact.group_id,
                "fold_id": artifact.fold_id,
                "split": "calibration_in_group",
                **compute_metrics(y_true_cal, y_pred_cal),
            }
            per_model_rows.append(row_cal)

        ecos_val = sorted(
            df_eval.loc[mask_oos, "ECO_ID"].dropna().astype(int).unique().tolist()
        )
        ecos_cal = sorted(
            df_eval.loc[mask_cal, "ECO_ID"].dropna().astype(int).unique().tolist()
        )
        split_rows.append(
            {
                "trait": trait,
                "study_name": artifact.study_name,
                "group_id": artifact.group_id,
                "fold_id": artifact.fold_id,
                "n_validation_samples": int(mask_oos.sum()),
                "n_calibration_samples": int(mask_cal.sum()),
                "n_validation_ecoregions": len(ecos_val),
                "n_calibration_ecoregions": len(ecos_cal),
                "validation_ecoregions": ";".join(map(str, ecos_val)),
                "calibration_ecoregions": ";".join(map(str, ecos_cal)),
            }
        )

    if not ensemble_preds:
        raise RuntimeError("No models were loaded for ensemble evaluation")

    ensemble_matrix = np.column_stack(ensemble_preds)
    y_pred_ensemble = ensemble_matrix.mean(axis=1)

    stacked_df = pd.DataFrame(
        {
            "uuid": stacked_uuid,
            "study_name": stacked_study,
            "y_true": np.concatenate(stacked_y_true)
            if stacked_y_true
            else np.array([]),
            "y_pred": np.concatenate(stacked_y_pred)
            if stacked_y_pred
            else np.array([]),
        }
    )

    ensemble_metrics = compute_metrics(y_all, y_pred_ensemble)
    summary_row: dict[str, float | str] = {
        "trait": trait,
        "n_models": float(ensemble_matrix.shape[1]),
        "rmse_ensemble": ensemble_metrics["rmse"],
        "mae_ensemble": ensemble_metrics["mae"],
        "r2_ensemble": ensemble_metrics["r2"],
        "bias_ensemble": ensemble_metrics["bias"],
        "n_ensemble": ensemble_metrics["n"],
        "rmse_stacked_oos": np.nan,
        "mae_stacked_oos": np.nan,
        "r2_stacked_oos": np.nan,
        "bias_stacked_oos": np.nan,
        "n_stacked_oos": np.nan,
    }

    if not stacked_df.empty:
        stacked_metrics = compute_metrics(
            stacked_df["y_true"].to_numpy(),
            stacked_df["y_pred"].to_numpy(),
        )
        summary_row.update(
            {
                "rmse_stacked_oos": stacked_metrics["rmse"],
                "mae_stacked_oos": stacked_metrics["mae"],
                "r2_stacked_oos": stacked_metrics["r2"],
                "bias_stacked_oos": stacked_metrics["bias"],
                "n_stacked_oos": stacked_metrics["n"],
            }
        )

    ensemble_predictions_df = df_eval[
        ["uuid", "ECO_ID", "RECODED_GROUP", "test_fold", trait]
    ].copy()
    ensemble_predictions_df = ensemble_predictions_df.rename(columns={trait: "y_true"})
    ensemble_predictions_df["y_pred_ensemble"] = y_pred_ensemble

    per_model_df = pd.DataFrame(per_model_rows)
    split_df = pd.DataFrame(split_rows)
    summary_df = pd.DataFrame([summary_row])
    return per_model_df, split_df, stacked_df, summary_df, ensemble_predictions_df


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir / args.trait
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_model_artifacts(args.models_dir, args.trait)
    logger.info(f"Loaded {len(artifacts)} model artifacts for trait='{args.trait}'")

    df_eval = load_eval_dataframe(args.cv_splits, args.trait)
    logger.info(f"Validation rows available for evaluation: {len(df_eval)}")

    per_model_df, split_df, stacked_df, summary_df, ensemble_predictions_df = (
        evaluate_models(
            df_eval=df_eval,
            artifacts=artifacts,
            trait=args.trait,
        )
    )

    per_model_path = output_dir / "per_model_metrics.csv"
    split_path = output_dir / "split_ecoregion_summary.csv"
    stacked_path = output_dir / "stacked_oos_predictions.csv"
    summary_path = output_dir / "summary_metrics.csv"
    ensemble_path = output_dir / "ensemble_predictions.csv"

    per_model_df.to_csv(per_model_path, index=False)
    split_df.to_csv(split_path, index=False)
    stacked_df.to_csv(stacked_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    ensemble_predictions_df.to_csv(ensemble_path, index=False)

    logger.info(f"Saved per-model metrics to {per_model_path}")
    logger.info(f"Saved split/ecoregion summary to {split_path}")
    logger.info(f"Saved stacked OOS predictions to {stacked_path}")
    logger.info(f"Saved ensemble predictions to {ensemble_path}")
    logger.info(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
