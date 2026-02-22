"""
NDVI-baseline benchmark vs Hybrid (S2Biophys_MLP) using Ecoregion-Based Spatially Blocked CV

Outputs (relative to this script's location by default):
  figures/ndvi_baseline_vs_hybrid_3x2_stacked_oof.png
  results/metrics_stacked_oof.csv
  results/predictions_stacked_oof.csv
  results/mae_by_ndvi_bin_oof.csv

This addresses reviewer comment:
- benchmark a naive NDVI→trait empirical curve (poly regression)
    against the hybrid PROSAIL-MLP approach, to clarify whether the added
    complexity provides a meaningful performance improvement over a much simpler approach.

Implementation:
- Third-degree polynomial NDVI–variable regression
- Fitted to GROUNDED-EO reference measurements
- Uses same ecoregion-based spatially blocked CV splits as S2BIOPHYS training
- Ensures fair comparison using identical train/validation split structure
- Demonstrates S2BIOPHYS advantage in physical consistency and extrapolation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from train_pipeline.finalTraining import (
    load_model_ensemble as load_specker_model_ensemble,
)
from train_pipeline.utilsLoading import load_grounded_eo_validation_data

TRAITS = ["laie", "fapar", "fcover"]


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_dirs(base_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Returns (figures_dir, results_dir).
    If base_dir is None: uses the directory of this file.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    figures_dir = ensure_dir(base_dir / "figures")
    results_dir = ensure_dir(base_dir / "results")
    return figures_dir, results_dir


# -----------------------------
# Core methods
# -----------------------------
def add_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    b8 = df["B8"] / 10000
    b4 = df["B4"] / 10000
    df["ndvi"] = (b8 - b4) / (b8 + b4)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def fit_ndvi_poly_baseline(x_train: np.ndarray, y_train: np.ndarray, degree: int = 3):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x_train, y_train)
    return model


def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def binned_mae(ndvi: np.ndarray, err: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"ndvi": ndvi, "err": err})
    df["bin"] = pd.cut(df["ndvi"], bins=n_bins, include_lowest=True)
    out = df.groupby("bin", observed=True)["err"].mean().reset_index()
    out["bin_mid"] = out["bin"].apply(lambda x: (x.left + x.right) / 2)
    return out


def binned_mean_curve(
    ndvi: np.ndarray, y: np.ndarray, n_bins: int = 25
) -> pd.DataFrame:
    df = pd.DataFrame({"ndvi": ndvi, "y": y})
    df["bin"] = pd.cut(df["ndvi"], bins=n_bins, include_lowest=True)
    out = df.groupby("bin", observed=True)["y"].mean().reset_index()
    out["bin_mid"] = out["bin"].apply(lambda x: (x.left + x.right) / 2)
    return out


def fit_ndvi_ensemble_with_cv_splits(
    df: pd.DataFrame,
    trait: str,
    s2biophys_models: dict,
    poly_degree: int = 3,
) -> dict:
    """
    Fit NDVI polynomial models using the same CV splits as S2BIOPHYS ensemble.

    Parameters
    ----------
    df : pd.DataFrame
        Validation data with 'ndvi', trait columns, and 'ECO_ID'.
    trait : str
        Trait name (e.g., 'laie', 'fapar', 'fcover').
    s2biophys_models : dict
        S2BIOPHYS model ensemble dictionary from load_specker_model_ensemble().
    poly_degree : int
        Polynomial degree for NDVI model.

    Returns
    -------
    dict
        Dictionary of fitted NDVI models, keyed by model name.
    """
    ndvi_models = {}

    for model_name, model_info in s2biophys_models.items():
        # Get training ecoregions for this fold
        train_ecos = model_info["split"]["val_ecos_train"]

        # Get training indices
        train_idx = df.index[df["ECO_ID"].isin(train_ecos)]

        # Get training data
        X_train = df.loc[train_idx, "ndvi"].to_numpy().reshape(-1, 1)
        y_train = df.loc[train_idx, trait].to_numpy()

        # Remove NaNs
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[mask]
        y_train = y_train[mask]

        # Fit model
        model = fit_ndvi_poly_baseline(X_train, y_train, degree=poly_degree)
        ndvi_models[model_name] = {
            "model": model,
            "split": model_info["split"],
        }

    return ndvi_models


def predict_stacked_oof(
    df: pd.DataFrame,
    trait: str,
    ndvi_models: dict,
    s2biophys_models: dict,
) -> pd.DataFrame:
    """
    Generate stacked out-of-fold predictions for both NDVI and S2BIOPHYS.

    Each sample is predicted by the single model where it was held out during training.

    Parameters
    ----------
    df : pd.DataFrame
        Validation data with features and 'ECO_ID'.
    trait : str
        Trait name.
    ndvi_models : dict
        NDVI model ensemble from fit_ndvi_ensemble_with_cv_splits().
    s2biophys_models : dict
        S2BIOPHYS model ensemble from load_specker_model_ensemble().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: index (original), y_true, yhat_ndvi_oof, yhat_s2biophys_oof, ndvi, ECO_ID.
    """
    # Prepare band features for S2BIOPHYS
    band_order = [
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

    df_prep = df.copy()

    # Rename angle columns if needed
    rename_map = {}
    if "sza" in df_prep.columns:
        rename_map["sza"] = "tts"
    if "vza" in df_prep.columns:
        rename_map["vza"] = "tto"
    if "phi" in df_prep.columns:
        rename_map["phi"] = "psi"
    df_prep = df_prep.rename(columns=rename_map)

    # Scale bands
    band_cols = [
        c
        for c in df_prep.columns
        if c.startswith("B") and (c[1:].replace("A", "").isdigit())
    ]
    df_prep[band_cols] = df_prep[band_cols] / 10000.0

    # Initialize results
    results = []

    # Loop through each fold
    for model_name in ndvi_models.keys():
        # Get test ecoregions for this fold
        test_ecos = ndvi_models[model_name]["split"]["val_ecos_test"]
        test_idx = df.index[df["ECO_ID"].isin(test_ecos)]

        if len(test_idx) == 0:
            continue

        # === NDVI Predictions ===
        X_ndvi_test = df_prep.loc[test_idx, "ndvi"].to_numpy().reshape(-1, 1)
        ndvi_model = ndvi_models[model_name]["model"]
        yhat_ndvi = ndvi_model.predict(X_ndvi_test)

        # === S2BIOPHYS Predictions ===
        X_s2_test = df_prep.loc[test_idx, band_order]
        s2_pipeline = s2biophys_models[model_name]["pipeline"]
        yhat_s2 = s2_pipeline.predict(X_s2_test).reshape(-1)

        # Clip predictions to valid ranges
        if trait == "laie":
            yhat_ndvi = np.clip(yhat_ndvi, 0, 8)
            yhat_s2 = np.clip(yhat_s2, 0, 8)
        else:  # fapar, fcover
            yhat_ndvi = np.clip(yhat_ndvi, 0, 1)
            yhat_s2 = np.clip(yhat_s2, 0, 1)

        # Store results
        for i, idx in enumerate(test_idx):
            results.append(
                {
                    "index": idx,
                    "y_true": df.loc[idx, trait],
                    "yhat_ndvi_oof": yhat_ndvi[i],
                    "yhat_s2biophys_oof": yhat_s2[i],
                    "ndvi": df.loc[idx, "ndvi"],
                    "ECO_ID": df.loc[idx, "ECO_ID"],
                }
            )

    return pd.DataFrame(results)


# -----------------------------
# Plotting
# -----------------------------
def make_simple_comparison(
    preds: pd.DataFrame,
    metrics_table: pd.DataFrame,
    plot_fraction: Optional[float] = None,
) -> plt.Figure:
    """
    3x2 panel comparing NDVI baseline vs S2BIOPHYS (both using stacked OOF predictions):
      rows: traits (LAI, fAPAR, fCOVER)
      col1: observed vs predicted (NDVI OOF + S2BIOPHYS OOF)
      col2: NDVI predictions vs S2BIOPHYS predictions
    """
    if plot_fraction is not None:
        preds = preds.sample(frac=plot_fraction, random_state=42)

    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(12, 10), constrained_layout=True
    )

    for i, trait in enumerate(TRAITS):
        d = preds[preds["trait"] == trait].copy()

        # -------- col1: Observed vs Predicted --------
        ax = axes[i, 0]
        ax.scatter(
            d["y_true"], d["yhat_ndvi_oof"], s=10, alpha=0.4, label="NDVI poly(3) OOF"
        )
        ax.scatter(
            d["y_true"],
            d["yhat_s2biophys_oof"],
            s=10,
            alpha=0.4,
            label="S2Biophys_MLP OOF",
        )

        lo = np.nanmin(
            [d["y_true"].min(), d["yhat_ndvi_oof"].min(), d["yhat_s2biophys_oof"].min()]
        )
        hi = np.nanmax(
            [d["y_true"].max(), d["yhat_ndvi_oof"].max(), d["yhat_s2biophys_oof"].max()]
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black")

        ax.set_xlabel("Observed (GROUNDED-EO)")
        ax.set_ylabel("Predicted")

        m_base = metrics_table[
            (metrics_table["trait"] == trait)
            & (metrics_table["model"] == "NDVI_poly3_OOF")
        ].iloc[0]
        m_hyb = metrics_table[
            (metrics_table["trait"] == trait)
            & (metrics_table["model"] == "S2Biophys_MLP_OOF")
        ].iloc[0]
        txt = (
            f"NDVI:      R²={m_base['r2']:.3f}, RMSE={m_base['rmse']:.3f}\n"
            f"S2Biophys: R²={m_hyb['r2']:.3f}, RMSE={m_hyb['rmse']:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        ax.set_title(f"{trait.upper()}: Observed vs Predicted")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        # -------- col2: NDVI Predictions vs S2BIOPHYS Predictions --------
        ax = axes[i, 1]
        ax.scatter(
            d["yhat_ndvi_oof"],
            d["yhat_s2biophys_oof"],
            s=10,
            alpha=0.4,
            c=d["y_true"],
            cmap="viridis",
        )

        lo_ndvi = d["yhat_ndvi_oof"].min()
        hi_ndvi = d["yhat_ndvi_oof"].max()
        lo_s2 = d["yhat_s2biophys_oof"].min()
        hi_s2 = d["yhat_s2biophys_oof"].max()

        lo = min(lo_ndvi, lo_s2)
        hi = max(hi_ndvi, hi_s2)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="red")

        ax.set_xlabel("NDVI Predictions (OOF)")
        ax.set_ylabel("S2Biophys Predictions (OOF)")
        ax.set_title(f"{trait.upper()}: Model Comparison")
        ax.grid(True, alpha=0.3)

    return fig


# -----------------------------
# Main
# -----------------------------
def main(
    base_dir: str | None = None,
    poly_degree: int = 3,
    n_bins_mae: int = 10,
    show: bool = True,
):
    """
    Compare NDVI baseline vs S2BIOPHYS using stacked out-of-fold (OOF) predictions.

    Parameters
    ----------
    base_dir : str or None
        Output base directory. If None, uses directory of this script.
        You can set env var NDVI_BENCHMARK_OUTDIR to override without editing code.
    poly_degree : int
        Polynomial degree for NDVI model.
    n_bins_mae : int
        Number of bins for MAE by NDVI bin analysis (for supplement table).
    show : bool
        Whether to display the plot.
    """
    # allow env override
    if base_dir is None:
        base_dir = os.getenv("NDVI_BENCHMARK_OUTDIR", None)

    figures_dir, results_dir = get_output_dirs(Path(base_dir) if base_dir else None)

    val = load_grounded_eo_validation_data()
    val = add_ndvi(val).dropna(subset=["ndvi", *TRAITS])

    print("\n=== NDVI Baseline vs S2BIOPHYS (Stacked OOF Comparison) ===")
    print(f"Total validation samples: {len(val)}")
    print(f"Polynomial degree: {poly_degree}")
    print(f"Traits: {TRAITS}\n")

    metrics_rows = []
    pred_rows = []

    for trait in TRAITS:
        print(f"Processing trait: {trait.upper()}")

        # Load S2BIOPHYS ensemble models for this trait
        s2biophys_models, _ = load_specker_model_ensemble(
            trait=trait, ensemble_size=5, model_version="v02"
        )
        print(f"  Loaded {len(s2biophys_models)} S2BIOPHYS models")

        # Fit NDVI models using the same CV splits
        ndvi_models = fit_ndvi_ensemble_with_cv_splits(
            df=val,
            trait=trait,
            s2biophys_models=s2biophys_models,
            poly_degree=poly_degree,
        )
        print(f"  Fitted {len(ndvi_models)} NDVI poly({poly_degree}) models")

        # Generate stacked OOF predictions
        oof_preds = predict_stacked_oof(
            df=val,
            trait=trait,
            ndvi_models=ndvi_models,
            s2biophys_models=s2biophys_models,
        )
        print(f"  Generated {len(oof_preds)} stacked OOF predictions")

        # Compute metrics
        ndvi_metrics = compute_metrics(
            oof_preds["y_true"].to_numpy(),
            oof_preds["yhat_ndvi_oof"].to_numpy(),
        )
        s2_metrics = compute_metrics(
            oof_preds["y_true"].to_numpy(),
            oof_preds["yhat_s2biophys_oof"].to_numpy(),
        )

        metrics_rows += [
            {"trait": trait, "model": "NDVI_poly3_OOF", **ndvi_metrics},
            {"trait": trait, "model": "S2Biophys_MLP_OOF", **s2_metrics},
        ]

        # Add trait column and compute errors for predictions table
        oof_preds["trait"] = trait
        oof_preds["err_ndvi_oof"] = np.abs(
            oof_preds["y_true"] - oof_preds["yhat_ndvi_oof"]
        )
        oof_preds["err_s2biophys_oof"] = np.abs(
            oof_preds["y_true"] - oof_preds["yhat_s2biophys_oof"]
        )

        pred_rows.append(oof_preds)
        print()

    metrics = pd.DataFrame(metrics_rows)
    preds = pd.concat(pred_rows, ignore_index=True)

    # --- Save tables ---
    metrics_csv = results_dir / "metrics_stacked_oof.csv"
    preds_csv = results_dir / "predictions_stacked_oof.csv"
    metrics.to_csv(metrics_csv, index=False)
    preds.to_csv(preds_csv, index=False)

    # also save MAE-by-NDVI-bin table (useful for supplement / reviewer)
    binned_rows = []
    for trait in TRAITS:
        d = preds[preds["trait"] == trait].copy()
        b0 = binned_mae(
            d["ndvi"].to_numpy(), d["err_ndvi_oof"].to_numpy(), n_bins=n_bins_mae
        )
        b1 = binned_mae(
            d["ndvi"].to_numpy(), d["err_s2biophys_oof"].to_numpy(), n_bins=n_bins_mae
        )
        b = b0[["bin", "bin_mid"]].copy()
        b["trait"] = trait
        b["mae_ndvi_oof"] = b0["err"].to_numpy()
        b["mae_s2biophys_oof"] = b1["err"].to_numpy()
        b["delta_mae_s2biophys_minus_ndvi"] = b["mae_s2biophys_oof"] - b["mae_ndvi_oof"]
        binned_rows.append(b)

    mae_by_bin = pd.concat(binned_rows, ignore_index=True)
    mae_csv = results_dir / "mae_by_ndvi_bin_oof.csv"
    mae_by_bin.to_csv(mae_csv, index=False)

    # --- Figure ---
    fig = make_simple_comparison(
        preds=preds,
        metrics_table=metrics,
        plot_fraction=0.2,
    )

    fig_path = figures_dir / "ndvi_baseline_vs_hybrid_3x2_stacked_oof.png"
    fig.savefig(fig_path, dpi=300)

    # console summary
    print("\n=== Stacked OOF Metrics (Fair Comparison) ===")
    print(
        metrics.pivot(
            index="trait", columns="model", values=["r2", "rmse", "mae"]
        ).round(3)
    )
    print("\nSaved:")
    print(f"  Figure:  {fig_path}")
    print(f"  Tables:  {metrics_csv}, {preds_csv}, {mae_csv}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show=False)
