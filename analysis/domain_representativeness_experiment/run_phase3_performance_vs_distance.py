"""
Phase 3.2: Performance as a Function of Distance to Training Domain
Generates S2BIOPHYS predictions for each RMs point, computes uncertainty,
and analyzes performance degradation with increasing distance to LUT.

Outputs:
- predictions_rms_phase3.csv: RMs predictions, errors, distances, uncertainty
- phase3_performance_vs_distance.png: Performance bins by distance
- phase3_uncertainty_vs_distance.png: Uncertainty vs distance
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_pipeline.finalTraining import predict_s2biophys
from train_pipeline.utilsLoading import load_grounded_eo_validation_data

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = "analysis/domain_representativeness_experiment"
DATA_DIR = f"{BASE_DIR}/data"
FIGURES_DIR = f"{BASE_DIR}/figures"
RESULTS_DIR = f"{BASE_DIR}/results"
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_ALL_FEATURES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "tts",
    "tto",
    "psi",
]
TRAITS = ["laie", "fapar", "fcover"]


def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepare_rms_features():
    """
    Load RMs data and prepare features for S2BIOPHYS prediction.
    Returns dataframe with bands scaled and angles renamed.
    """
    print("Loading RMs validation data...")
    val = load_grounded_eo_validation_data()

    df_prep = val.copy()

    # Rename angle columns if needed
    rename_map = {}
    if "sza" in df_prep.columns:
        rename_map["sza"] = "tts"
    if "vza" in df_prep.columns:
        rename_map["vza"] = "tto"
    if "phi" in df_prep.columns:
        rename_map["phi"] = "psi"
    df_prep = df_prep.rename(columns=rename_map)

    # Scale bands to reflectance (0-1)
    band_cols = [
        c
        for c in df_prep.columns
        if c.startswith("B") and (c[1:].replace("A", "").isdigit())
    ]
    df_prep[band_cols] = df_prep[band_cols] / 10000.0

    print(f"  RMs samples: {len(df_prep)}")
    print(f"  Columns available: {df_prep.columns.tolist()}")

    return df_prep


def load_rms_distances():
    """Load pre-computed distances from Phase 3.1."""
    print("Loading Phase 3.1 distance data...")

    rms_dist_file = Path(f"{RESULTS_DIR}/phase3_rms_with_distances.csv")
    if not rms_dist_file.exists():
        raise FileNotFoundError(
            f"Phase 3.1 output not found: {rms_dist_file}\n"
            "Please run run_phase3_distance_analysis.py first."
        )

    rms_dist = pd.read_csv(rms_dist_file)
    print(f"  Loaded distances for {len(rms_dist)} RMs points")
    return rms_dist


def predict_rms_ensemble(df_rms, ensemble_size: int = 5):
    """
    Generate full S2BIOPHYS ensemble predictions for all RMs points.

    For FAPAR/FCOVER
    """

    pred_df = predict_s2biophys(
        df_rms,
        {"ensemble_size": ensemble_size, "model_version": "v02"},
        recalibrate_uncertainty=True,
    )

    return pred_df


def compute_ensemble_metrics(predictions_ensemble, y_true, trait: str):
    """
    Compute mean predictions, ensemble uncertainty, and errors.

    Returns:
    --------
    dict with keys:
        yhat_mean: ensemble mean prediction
        yhat_std: ensemble std (uncertainty)
        error_abs: absolute error
        error_signed: signed error
    """
    # Clip predictions to valid ranges
    if trait == "laie":
        predictions_ensemble = np.clip(predictions_ensemble, 0, 8)
    else:  # fapar, fcover
        predictions_ensemble = np.clip(predictions_ensemble, 0, 1)

    yhat_mean = np.mean(predictions_ensemble, axis=1)
    yhat_std = np.std(predictions_ensemble, axis=1)
    error_abs = np.abs(y_true - yhat_mean)
    error_signed = y_true - yhat_mean

    return {
        "yhat_mean": yhat_mean,
        "yhat_std": yhat_std,
        "error_abs": error_abs,
        "error_signed": error_signed,
    }


def compute_global_metrics(y_true, yhat_mean):
    """Compute R2, RMSE, MAE."""
    return {
        "r2": float(r2_score(y_true, yhat_mean)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, yhat_mean))),
        "mae": float(mean_absolute_error(y_true, yhat_mean)),
    }


def bin_by_distance(df_predictions, trait: str, bin_size: int = 200):
    """
    Bin predictions by distance to LUT and compute performance metrics.

    Returns:
    --------
    pd.DataFrame with columns:
        distance_bin_lo, distance_bin_hi, distance_bin_mid,
        n_samples, mae, rmse, r2, mean_uncertainty
    """

    y_true_col = trait
    yhat_col = f"s2biophys_{trait}_mean"
    ystd_col = f"s2biophys_{trait}_std"

    d = df_predictions[[y_true_col, yhat_col, ystd_col, "distance_to_lut"]].dropna()

    # Create equal-count bins by sorting and chunking
    d = d.sort_values("distance_to_lut").reset_index(drop=True)
    d["distance_bin"] = (np.arange(len(d)) // bin_size).astype(int)

    results = []
    for bin_id, group in d.groupby("distance_bin", observed=True):
        bin_lo = group["distance_to_lut"].min()
        bin_hi = group["distance_to_lut"].max()
        bin_mid = group["distance_to_lut"].median()

        mae = mean_absolute_error(group[y_true_col], group[yhat_col])
        rmse = np.sqrt(mean_squared_error(group[y_true_col], group[yhat_col]))
        r2 = r2_score(group[y_true_col], group[yhat_col]) if len(group) > 1 else np.nan
        mean_unc = group[ystd_col].mean()

        results.append(
            {
                "trait": trait,
                "distance_bin_lo": bin_lo,
                "distance_bin_hi": bin_hi,
                "distance_bin_mid": bin_mid,
                "n_samples": len(group),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mean_uncertainty": mean_unc,
            }
        )

    return pd.DataFrame(results)


def plot_performance_vs_distance(df_predictions, output_dir: Path):
    """Plot performance metrics as a function of distance to LUT."""
    print("\nGenerating performance vs distance plots...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Performance as a Function of Distance to LUT Training Domain",
        fontsize=14,
        fontweight="bold",
    )

    for idx, trait in enumerate(TRAITS):
        trait_col = trait
        yhat_col = f"s2biophys_{trait}_mean"
        yhat_std_col = f"s2biophys_{trait}_std"

        d = df_predictions[
            [trait_col, yhat_col, yhat_std_col, "distance_to_lut"]
        ].dropna()

        d["error_abs"] = np.abs(d[trait_col] - d[yhat_col])

        # Bin by distance (equal-count bins)
        binned = bin_by_distance(d, trait=trait, bin_size=400)

        ax = axes[idx]

        # Plot RMSE vs distance (converted to actual distance values)
        ax.plot(
            binned["distance_bin_mid"],
            binned["rmse"],
            "o-",
            markersize=8,
            linewidth=2,
            alpha=0.7,
            label="RMSE",
        )

        ax.set_xlabel("Distance to LUT", fontsize=11)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.set_title(f"{trait.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Plot R2 on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            binned["distance_bin_mid"],
            binned["r2"],
            "s--",
            markersize=6,
            linewidth=1.8,
            alpha=0.7,
            color="tab:orange",
            label="R2",
        )
        ax2.set_ylabel("R2", fontsize=11)

        # Combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path = output_dir / "phase3_performance_vs_distance.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def plot_uncertainty_vs_distance(df_predictions, output_dir: Path):
    """Plot ensemble uncertainty as a function of distance to LUT."""
    print("Generating uncertainty vs distance plots...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Ensemble Uncertainty as a Function of Distance to LUT Training Domain",
        fontsize=14,
        fontweight="bold",
    )

    for idx, trait in enumerate(TRAITS):
        trait_col = trait
        yhat_col = f"s2biophys_{trait}_mean"
        yhat_std_col = f"s2biophys_{trait}_std"

        d = df_predictions[
            [trait_col, yhat_col, yhat_std_col, "distance_to_lut"]
        ].dropna()

        d["error_abs"] = np.abs(d[trait_col] - d[yhat_col])

        ax = axes[idx]

        # Bin-wise mean (equal-count bins) on bin index
        binned = bin_by_distance(d, trait=trait, bin_size=400)
        bin_idx = np.arange(1, len(binned) + 1)
        ax.plot(
            bin_idx,
            binned["mean_uncertainty"],
            "r-",
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="Binned mean uncertainty",
        )

        # Boxplot per bin (all points in interval)
        d_sorted = d.sort_values("distance_to_lut").reset_index(drop=True)
        d_sorted["distance_bin"] = (np.arange(len(d_sorted)) // 400).astype(int)
        box_data = [
            grp[yhat_std_col].values
            for _, grp in d_sorted.groupby("distance_bin", observed=True)
        ]
        ax.boxplot(
            box_data,
            positions=bin_idx,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9d9d9", "alpha": 0.6},
            medianprops={"color": "#333333", "linewidth": 1.2},
            whiskerprops={"color": "#666666"},
            capprops={"color": "#666666"},
        )

        ax.set_xlabel("Distance bins (sorted by distance)", fontsize=11)
        ax.set_xticks([])
        ax.set_ylabel("Ensemble Std (Uncertainty)", fontsize=11)
        ax.set_title(f"{trait.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = output_dir / "phase3_uncertainty_vs_distance.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()


def main():
    """Run Phase 3.2 analysis."""
    ensure_dir(Path(FIGURES_DIR))
    ensure_dir(Path(RESULTS_DIR))

    # Load RMs features
    df_rms = prepare_rms_features()

    # predict s2biophys
    preds_s2biophys = predict_s2biophys(
        df_rms,
        {"ensemble_size": 5, "model_version": "v02"},
        recalibrate_uncertainty=True,
        clip_predictions=True,
    )

    # Load pre-computed distances from Phase 3.1
    rms_distances = load_rms_distances()

    # merge all three on uuid
    df_rms = df_rms.merge(
        rms_distances[["uuid", "distance_to_lut"]],
        on="uuid",
        how="left",
    )

    df_all = df_rms.merge(
        preds_s2biophys,
        on="uuid",
        how="left",
    )

    # true values are e.g. 'laie' ; predictions 's2biophys_laie_mean', 's2biophys_laie_std'; distance to LUT: 'distance_to_lut'
    # Save predictions table
    output_csv = Path(RESULTS_DIR) / "predictions_rms_phase3.csv"
    df_all.to_csv(output_csv, index=False)
    print(f"\nSaved predictions: {output_csv}")

    # Generate binned metrics for each trait
    print("\n" + "=" * 70)
    print("BINNED PERFORMANCE METRICS BY DISTANCE")
    print("=" * 70)

    binned_results = []
    for trait in TRAITS:
        binned = bin_by_distance(df_all, trait=trait, bin_size=200)
        binned_results.append(binned)
        print(f"\n{trait.upper()}:")
        print(binned.to_string(index=False))

    binned_all = pd.concat(binned_results, ignore_index=True)
    binned_csv = Path(RESULTS_DIR) / "performance_by_distance_phase3.csv"
    binned_all.to_csv(binned_csv, index=False)
    print(f"\nSaved binned metrics: {binned_csv}")

    # Plots (using RMSE increase thresholds for visualization)
    plot_performance_vs_distance(df_all, Path(FIGURES_DIR))
    plot_uncertainty_vs_distance(df_all, Path(FIGURES_DIR))

    print("\n" + "=" * 70)
    print("PHASE 3.2 COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    print(f"  {output_csv}")
    print(f"  {binned_csv}")
    print(f"  {FIGURES_DIR}/phase3_performance_vs_distance.png")
    print(f"  {FIGURES_DIR}/phase3_uncertainty_vs_distance.png")


if __name__ == "__main__":
    main()
