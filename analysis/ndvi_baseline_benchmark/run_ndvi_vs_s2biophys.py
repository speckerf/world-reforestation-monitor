"""
NDVI-baseline benchmark vs Hybrid (S2Biophys_MLP)

Outputs (relative to this script's location by default):
  figures/ndvi_baseline_vs_hybrid_3x3.png
  results/metrics_testset.csv
  results/predictions_testset.csv
  results/mae_by_ndvi_bin.csv

This addresses reviewer comment:
- benchmark a naive NDVI→trait empirical curve (poly regression)
    against the hybrid PROSAIL-MLP approach, to clarify whether the added complexity provides a meaningful performance improvement over a much simpler approach.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from config.config import get_config
from train_pipeline.finalTraining import predict_s2biophys
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


# -----------------------------
# Plotting
# -----------------------------
def make_3x3_panel(
    preds: pd.DataFrame,
    baseline_models: dict,
    metrics_table: pd.DataFrame,
    poly_degree: int = 3,
    n_bins_curve: int = 25,
    n_bins_mae: int = 10,
) -> plt.Figure:
    """
    3x3 panel:
      rows: traits
      col1: NDVI vs observed + baseline curve + hybrid binned mean curve
      col2: observed vs predicted (baseline + hybrid)
      col3: MAE by NDVI bin (baseline + hybrid)
    """
    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(13.5, 11), constrained_layout=True
    )

    for i, trait in enumerate(TRAITS):
        d = preds[preds["trait"] == trait].copy()

        # -------- col1: NDVI vs observed + curves --------
        ax = axes[i, 0]
        ax.scatter(
            d["ndvi"], d["y_true"], s=6, alpha=0.25, label="GROUNDED-EO obs (test)"
        )

        model = baseline_models[trait]
        ndvi_grid = np.linspace(np.nanmin(d["ndvi"]), np.nanmax(d["ndvi"]), 250)
        y_base_curve = model.predict(ndvi_grid.reshape(-1, 1))
        ax.plot(
            ndvi_grid, y_base_curve, linewidth=2, label=f"NDVI poly({poly_degree}) fit"
        )

        hyb_curve = binned_mean_curve(
            d["ndvi"].to_numpy(),
            d["yhat_s2biophys"].to_numpy(),
            n_bins=n_bins_curve,
        )
        ax.plot(
            hyb_curve["bin_mid"],
            hyb_curve["y"],
            linewidth=2,
            label="S2Biophys_MLP (binned mean)",
        )

        ax.set_xlabel("NDVI")
        ax.set_ylabel(trait.upper())
        ax.set_title(f"{trait.upper()}: NDVI→Trait relationship")
        ax.legend(fontsize=8, loc="best")

        # -------- col2: Observed vs Predicted --------
        ax = axes[i, 1]
        ax.scatter(d["y_true"], d["yhat_base"], s=10, alpha=0.35, label="NDVI poly(3)")
        ax.scatter(
            d["y_true"],
            d["yhat_s2biophys"],
            s=10,
            alpha=0.35,
            label="S2Biophys_MLP",
        )

        lo = np.nanmin(
            [d["y_true"].min(), d["yhat_base"].min(), d["yhat_s2biophys"].min()]
        )
        hi = np.nanmax(
            [d["y_true"].max(), d["yhat_base"].max(), d["yhat_s2biophys"].max()]
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

        ax.set_xlabel("Observed (GROUNDED-EO test)")
        ax.set_ylabel("Predicted")

        m_base = metrics_table[
            (metrics_table["trait"] == trait) & (metrics_table["model"] == "NDVI_poly3")
        ].iloc[0]
        m_hyb = metrics_table[
            (metrics_table["trait"] == trait)
            & (metrics_table["model"] == "S2Biophys_MLP")
        ].iloc[0]
        txt = (
            f"NDVI poly3: R²={m_base['r2']:.2f}, RMSE={m_base['rmse']:.2f}\n"
            f"S2Biophys:  R²={m_hyb['r2']:.2f}, RMSE={m_hyb['rmse']:.2f}"
        )
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9
        )

        ax.set_title(f"{trait.upper()}: Observed vs Predicted")
        ax.legend(fontsize=8, loc="best")

        # -------- col3: MAE by NDVI bin --------
        ax = axes[i, 2]
        b_base = binned_mae(
            d["ndvi"].to_numpy(), d["err_base"].to_numpy(), n_bins=n_bins_mae
        )
        b_hyb = binned_mae(
            d["ndvi"].to_numpy(), d["err_s2biophys"].to_numpy(), n_bins=n_bins_mae
        )

        ax.plot(
            b_base["bin_mid"],
            b_base["err"],
            marker="o",
            linewidth=2,
            label="NDVI poly(3)",
        )
        ax.plot(
            b_hyb["bin_mid"],
            b_hyb["err"],
            marker="o",
            linewidth=2,
            label="S2Biophys_MLP",
        )

        ax.set_xlabel("NDVI (bin mid)")
        ax.set_ylabel("MAE")
        ax.set_title(f"{trait.upper()}: MAE by NDVI bin")
        ax.legend(fontsize=8, loc="best")

    return fig


# -----------------------------
# Main
# -----------------------------
def main(
    base_dir: str | None = None,
    poly_degree: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    n_bins_curve: int = 25,
    n_bins_mae: int = 10,
    show: bool = True,
):
    """
    base_dir: output base directory. If None, uses directory of this script.
              You can set env var NDVI_BENCHMARK_OUTDIR to override without editing code.
    """
    # allow env override
    if base_dir is None:
        base_dir = os.getenv("NDVI_BENCHMARK_OUTDIR", None)

    figures_dir, results_dir = get_output_dirs(Path(base_dir) if base_dir else None)

    config = get_config("train_pipeline")
    val = load_grounded_eo_validation_data()
    val = add_ndvi(val).dropna(subset=["ndvi", *TRAITS])

    idx = np.arange(len(val))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )

    X = val["ndvi"].to_numpy().reshape(-1, 1)

    # run s2biophys once
    s2pred = predict_s2biophys(df=val, config=config, recalibrate_uncertainty=False)

    metrics_rows = []
    pred_rows = []
    baseline_models = {}

    for trait in TRAITS:
        y = val[trait].to_numpy()

        # baseline (fit train)
        baseline = fit_ndvi_poly_baseline(
            X[idx_train], y[idx_train], degree=poly_degree
        )
        baseline_models[trait] = baseline
        yhat_base = baseline.predict(X[idx_test])

        # s2biophys (same test indices)
        s2biophys_col = f"s2biophys_{trait}_mean"
        if s2biophys_col not in s2pred.columns:
            raise KeyError(
                f"Missing column {s2biophys_col} in predict_s2biophys output."
            )
        yhat_s2 = s2pred.loc[val.index[idx_test], s2biophys_col].to_numpy()

        # metrics
        base_m = compute_metrics(y[idx_test], yhat_base)
        s2_m = compute_metrics(y[idx_test], yhat_s2)

        metrics_rows += [
            {"trait": trait, "model": "NDVI_poly3", **base_m},
            {"trait": trait, "model": "S2Biophys_MLP", **s2_m},
        ]

        # predictions table
        pred_rows.append(
            pd.DataFrame(
                {
                    "trait": trait,
                    "ndvi": val["ndvi"].to_numpy()[idx_test],
                    "y_true": y[idx_test],
                    "yhat_base": yhat_base,
                    "yhat_s2biophys": yhat_s2,
                    "err_base": np.abs(y[idx_test] - yhat_base),
                    "err_s2biophys": np.abs(y[idx_test] - yhat_s2),
                }
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    preds = pd.concat(pred_rows, ignore_index=True)

    # --- Save tables ---
    metrics_csv = results_dir / "metrics_testset.csv"
    preds_csv = results_dir / "predictions_testset.csv"
    metrics.to_csv(metrics_csv, index=False)
    preds.to_csv(preds_csv, index=False)

    # also save MAE-by-NDVI-bin table (useful for supplement / reviewer)
    binned_rows = []
    for trait in TRAITS:
        d = preds[preds["trait"] == trait].copy()
        b0 = binned_mae(
            d["ndvi"].to_numpy(), d["err_base"].to_numpy(), n_bins=n_bins_mae
        )
        b1 = binned_mae(
            d["ndvi"].to_numpy(), d["err_s2biophys"].to_numpy(), n_bins=n_bins_mae
        )
        b = b0[["bin", "bin_mid"]].copy()
        b["trait"] = trait
        b["mae_base"] = b0["err"].to_numpy()
        b["mae_s2biophys"] = b1["err"].to_numpy()
        b["delta_mae_s2biophys_minus_base"] = b["mae_s2biophys"] - b["mae_base"]
        binned_rows.append(b)

    mae_by_bin = pd.concat(binned_rows, ignore_index=True)
    mae_csv = results_dir / "mae_by_ndvi_bin.csv"
    mae_by_bin.to_csv(mae_csv, index=False)

    # --- Figure ---
    fig = make_3x3_panel(
        preds=preds,
        baseline_models=baseline_models,
        metrics_table=metrics,
        poly_degree=poly_degree,
        n_bins_curve=n_bins_curve,
        n_bins_mae=n_bins_mae,
    )

    fig_path = figures_dir / "ndvi_baseline_vs_hybrid_3x3.png"
    fig.savefig(fig_path, dpi=300)

    # console summary
    print("\n=== Test-set metrics (same split) ===")
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
