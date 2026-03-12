"""
Create publication-style combined figure for LAIe only:
Top left: Performance vs distance (RMSE + R2)
Top right: Uncertainty vs distance (mean + boxplot)
Bottom left: PCA domain plot
Bottom middle: Distance distribution densities
Bottom right: Global map placeholder
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from analysis.domain_representativeness_experiment.run_phase2_pca import (
    load_phase1_data,
    perform_2d_pca,
    subsample_points,
)
from analysis.domain_representativeness_experiment.run_phase3_performance_vs_distance import (
    bin_by_distance,
)
from train_pipeline.finalTraining import predict_s2biophys
from train_pipeline.utilsLoading import load_grounded_eo_validation_data

# Configuration
BASE_DIR = Path("analysis/domain_representativeness_experiment")
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# Focus only on LAIe
TRAIT = "fapar"
BIN_SIZE = 300  # Number of points per bin


def build_df_all() -> object:
    """Load RMs data, predict S2BIOPHYS, and merge with LUT distances."""
    df_rms = load_grounded_eo_validation_data()

    # Predict S2BIOPHYS with recalibrated uncertainty
    preds_s2biophys = predict_s2biophys(
        df_rms,
        {"ensemble_size": 5, "model_version": "v02"},
        recalibrate_uncertainty=True,
    )

    # Load pre-computed distances
    rms_dist_file = RESULTS_DIR / "phase3_rms_with_distances.csv"
    rms_distances = rms_dist_file
    if not rms_distances.exists():
        raise FileNotFoundError(
            f"Missing distances file: {rms_distances}. Run run_phase3_distance_analysis.py first."
        )
    rms_distances = np.genfromtxt(
        rms_distances, delimiter=",", names=True, dtype=None, encoding=None
    )

    # Convert distances to dict-like structure
    # Expect columns: uuid, distance_to_lut
    dist_uuid = rms_distances["uuid"] if "uuid" in rms_distances.dtype.names else None
    dist_vals = rms_distances["distance_to_lut"]

    # Merge by uuid
    if dist_uuid is None or "uuid" not in df_rms.columns:
        raise ValueError("UUID column missing in distances or RMs data.")

    df_rms = df_rms.merge(
        preds_s2biophys,
        on="uuid",
        how="left",
    )

    # Attach distances
    dist_map = {u: d for u, d in zip(dist_uuid, dist_vals)}
    df_rms["distance_to_lut"] = df_rms["uuid"].map(dist_map)

    return df_rms


def add_panel_label(ax, label: str) -> None:
    """Add panel label (e.g., A1) to the top-left of an axis."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )


def plot_pca_panel(ax) -> None:
    """Plot PCA domain panel for LAIe (bottom left)."""
    data = load_phase1_data(DATA_DIR)
    pca_results = perform_2d_pca(data)
    pca_results_sub = subsample_points(pca_results, n_samples=3000)

    lut_pca = pca_results_sub[TRAIT]["lut_pca"]
    rms_pca = pca_results_sub[TRAIT]["rms_pca"]
    s2_pca = pca_results_sub[TRAIT]["s2_pca"]
    pca = pca_results_sub[TRAIT]["pca"]

    ax.scatter(
        lut_pca[:, 0],
        lut_pca[:, 1],
        alpha=0.2,
        s=3,
        c="blue",
        label="PROSAIL LUT",
    )
    ax.scatter(
        rms_pca[:, 0],
        rms_pca[:, 1],
        alpha=0.2,
        s=3,
        c="red",
        label="Grounded-EO RMs",
        edgecolors="darkred",
        linewidth=0.3,
    )
    ax.scatter(
        s2_pca[:, 0],
        s2_pca[:, 1],
        alpha=0.2,
        s=3,
        c="green",
        label="Global S2",
        edgecolors="darkgreen",
        linewidth=0.3,
    )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        fontsize=10,
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        fontsize=10,
    )
    ax.set_title(
        f"Domain Representation - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_performance_panel(ax, df_all) -> None:
    """Plot performance (RMSE + R2) panel for LAIe (top left)."""
    binned = bin_by_distance(df_all, trait=TRAIT, bin_size=BIN_SIZE)

    ax.plot(
        binned["distance_bin_mid"],
        binned["rmse"],
        "o-",
        markersize=7,
        linewidth=2,
        alpha=0.8,
        label="RMSE",
        color="tab:blue",
    )
    ax.set_ylabel("RMSE", fontsize=10, color="tab:blue")
    ax.set_xlabel("Distance to LUT", fontsize=10)
    ax.set_title(
        f"Performance vs Distance - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(
        binned["distance_bin_mid"],
        binned["r2"],
        "s--",
        markersize=6,
        linewidth=2,
        alpha=0.8,
        color="tab:orange",
        label="R²",
    )
    ax2.set_ylabel("R²", fontsize=10, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add recommended cutoff line at distance = 3.5
    ax.axvline(
        3.5,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Recommended cutoff (3.5)",
    )

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=9, loc="upper right")


def plot_uncertainty_panel(ax, df_all) -> None:
    """Plot uncertainty (mean + boxplots) panel for LAIe (top right)."""
    binned = bin_by_distance(df_all, trait=TRAIT, bin_size=BIN_SIZE)
    bin_idx = np.arange(1, len(binned) + 1)

    ax.plot(
        bin_idx,
        binned["mean_uncertainty"],
        "r-",
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="Mean uncertainty",
    )

    # Boxplots
    d = df_all[[TRAIT, f"s2biophys_{TRAIT}_std", "distance_to_lut"]].dropna()
    d = d.sort_values("distance_to_lut").reset_index(drop=True)
    d["distance_bin"] = (np.arange(len(d)) // BIN_SIZE).astype(int)
    box_data = [
        grp[f"s2biophys_{TRAIT}_std"].values
        for _, grp in d.groupby("distance_bin", observed=True)
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

    # Add recommended cutoff line at distance = 3.5
    # Place line between third and second last bin
    cutoff_position = len(binned) - 1.5  # Between third-to-last and second-to-last bin
    ax.axvline(
        cutoff_position,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Recommended cutoff (3.5)",
    )

    ax.set_xlabel("Distance to LUT bins (sorted by distance)", fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Uncertainty (std)", fontsize=10)
    ax.set_title(
        f"Uncertainty vs Distance - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")


def plot_density_distributions_panel(ax) -> None:
    """Plot distance distribution densities (bottom middle)."""
    # Load the distance data from phase 3 results
    rms_dist_file = RESULTS_DIR / "phase3_rms_with_distances.csv"
    global_dist_file = RESULTS_DIR / "phase3_globalS2_with_distances.csv"

    if not rms_dist_file.exists() or not global_dist_file.exists():
        ax.text(
            0.5,
            0.5,
            "Distance data not available\n\nRun run_phase3_distance_analysis.py first",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            style="italic",
            color="red",
        )
        return

    # Load distance data
    rms_data = pd.read_csv(rms_dist_file)
    global_data = pd.read_csv(global_dist_file)

    rms_distances = rms_data["distance_to_lut"].values
    global_distances = global_data["distance_to_lut"].values

    # Create density plots similar to phase 3 analysis
    x_range = np.linspace(0, max(np.max(rms_distances), np.max(global_distances)), 200)

    # KDE for RMs
    kde_rms = gaussian_kde(rms_distances)
    ax.plot(x_range, kde_rms(x_range), "r-", linewidth=2.5, label="RMs", alpha=0.8)
    ax.fill_between(x_range, kde_rms(x_range), alpha=0.3, color="red")

    # KDE for Global S2
    kde_global = gaussian_kde(global_distances)
    ax.plot(
        x_range, kde_global(x_range), "g-", linewidth=2.5, label="Global S2", alpha=0.8
    )
    ax.fill_between(x_range, kde_global(x_range), alpha=0.3, color="green")

    # Add recommended cutoff line at distance = 3.5
    ax.axvline(
        3.5,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Recommended cutoff (3.5)",
    )

    ax.set_xlabel("Distance to LUT (standardized Manhattan)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Distance Distributions - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, 8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_global_map_placeholder_panel(ax) -> None:
    """Plot global map placeholder panel (bottom right)."""
    ax.text(
        0.5,
        0.5,
        "Global Map\n\n(Average distance to LUT\nat 100m resolution)",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
        style="italic",
        color="gray",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Global Distance to LUT", fontsize=11, fontweight="bold")

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor("lightgray")
        spine.set_linestyle("--")
        spine.set_linewidth(1)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build data
    df_all = build_df_all()

    # Create 3x2 layout with bottom panel spanning both columns
    fig = plt.figure(figsize=(12, 12))

    # Top row: Performance and uncertainty
    ax_a = plt.subplot(3, 2, 1)
    plot_performance_panel(ax_a, df_all)
    add_panel_label(ax_a, "A")

    ax_b = plt.subplot(3, 2, 2)
    plot_uncertainty_panel(ax_b, df_all)
    add_panel_label(ax_b, "B")

    # Middle row: PCA and density distributions
    ax_c = plt.subplot(3, 2, 3)
    plot_pca_panel(ax_c)
    add_panel_label(ax_c, "C")

    ax_d = plt.subplot(3, 2, 4)
    plot_density_distributions_panel(ax_d)
    add_panel_label(ax_d, "D")

    # Bottom row: Global map spanning both columns
    ax_e = plt.subplot(3, 1, 3)
    plot_global_map_placeholder_panel(ax_e)
    add_panel_label(ax_e, "E")

    plt.tight_layout()
    out_path = FIGURES_DIR / f"combined_domain_figure_{TRAIT}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined {TRAIT.upper()} figure: {out_path}")


if __name__ == "__main__":
    main()
