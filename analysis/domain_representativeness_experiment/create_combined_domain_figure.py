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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    # ax.set_title(
    #     f"Domain Representation - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    # )
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
    ax.set_xlabel("Distance to PROSAIL LUT", fontsize=10)
    # ax.set_title(
    #     f"Performance vs Distance - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    # )
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

    ax.set_xlabel("Distance to PROSAIL LUT (sorted bins)", fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Uncertainty (std)", fontsize=10)
    # ax.set_title(
    #     f"Uncertainty vs Distance - {TRAIT.upper()}", fontsize=11, fontweight="bold"
    # )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")


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
    # ax.set_title("Global Distance to LUT", fontsize=11, fontweight="bold")

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor("lightgray")
        spine.set_linestyle("--")
        spine.set_linewidth(1)


def plot_distance_cdf_comparison(ax) -> None:
    """Plot cumulative distribution function comparison for S2 distances."""
    # Load Global S2 distance data (to LUT)
    global_dist_file = RESULTS_DIR / "phase3_globalS2_with_distances.csv"

    if not global_dist_file.exists():
        ax.text(
            0.5,
            0.5,
            "Distance data not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            style="italic",
            color="red",
        )
        return

    try:
        from sklearn.neighbors import NearestNeighbors

        from analysis.domain_representativeness_experiment.run_phase2_pca import (
            load_phase1_data,
            perform_2d_pca,
        )

        # Load Global S2 distances to LUT
        global_data = pd.read_csv(global_dist_file)
        s2_to_lut_distances = global_data["distance_to_lut"].values

        # Compute S2 distances to RMs using same approach as panel A
        data = load_phase1_data(DATA_DIR)
        pca_results = perform_2d_pca(data)
        s2_pca = pca_results[TRAIT]["s2_pca"]
        rms_pca = pca_results[TRAIT]["rms_pca"]
        lut_pca = pca_results[TRAIT]["lut_pca"]

        # Use equal-sized subsamples of LUT and RMs for fair comparison
        min_ref_size = min(len(lut_pca), len(rms_pca))
        np.random.seed(42)

        # Subsample reference datasets to equal size
        lut_indices = np.random.choice(len(lut_pca), size=min_ref_size, replace=False)
        rms_indices = np.random.choice(len(rms_pca), size=min_ref_size, replace=False)

        lut_subset = lut_pca[lut_indices]
        rms_subset = rms_pca[rms_indices]

        # Calculate avg manhattan distance to 5 nearest neighbours
        nbrs_lut = NearestNeighbors(n_neighbors=5, metric="cityblock").fit(lut_subset)
        nbrs_rms = NearestNeighbors(n_neighbors=5, metric="cityblock").fit(rms_subset)

        # Calculate distances to nearest neighbors in the reference datasets
        dst_s2_to_lut, _ = nbrs_lut.kneighbors(s2_pca)
        dst_s2_to_rms, _ = nbrs_rms.kneighbors(s2_pca)

        # Also calculate RMs to LUT distances
        dst_rms_to_lut, _ = nbrs_lut.kneighbors(rms_subset)

        s2_to_lut_subset = dst_s2_to_lut.mean(axis=1)
        s2_to_rms_subset = dst_s2_to_rms.mean(axis=1)
        rms_to_lut_subset = dst_rms_to_lut.mean(axis=1)

        # Create CDFs
        x_lut = np.sort(s2_to_lut_subset)
        y_lut = np.arange(1, len(x_lut) + 1) / len(x_lut)

        x_rms = np.sort(s2_to_rms_subset)
        y_rms = np.arange(1, len(x_rms) + 1) / len(x_rms)

        x_rms_to_lut = np.sort(rms_to_lut_subset)
        y_rms_to_lut = np.arange(1, len(x_rms_to_lut) + 1) / len(x_rms_to_lut)

        ax.plot(x_lut, y_lut, color="blue", linewidth=2.5, label="S2 → PROSAIL LUT")
        ax.plot(x_rms, y_rms, color="red", linewidth=2.5, label="S2 → GROUNDED-EO RMs")
        ax.plot(
            x_rms_to_lut,
            y_rms_to_lut,
            color="orange",
            linewidth=2.5,
            label="GROUNDED-EO RMs → PROSAIL LUT",
        )
        ax.axvline(
            3.5,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Recommended cutoff (3.5)",
        )

        ax.set_xlabel("Distance to X", fontsize=10)
        ax.set_ylabel("Cumulative Probability", fontsize=10)
        # ax.set_title(
        #     f"Distance Comparison CDFs - {TRAIT.upper()}",
        #     fontsize=11,
        #     fontweight="bold",
        # )
        # ax.legend(fontsize=9)  # Legend removed
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 8)
    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            style="italic",
            color="red",
        )


def plot_distance_statistics_table(ax) -> None:
    """Plot statistical summary table for S2 distance comparisons."""
    # Load Global S2 distance data (to LUT)
    global_dist_file = RESULTS_DIR / "phase3_globalS2_with_distances.csv"

    if not global_dist_file.exists():
        ax.text(
            0.5,
            0.5,
            "Distance data not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            style="italic",
            color="red",
        )
        return

    try:
        from sklearn.neighbors import NearestNeighbors

        from analysis.domain_representativeness_experiment.run_phase2_pca import (
            load_phase1_data,
            perform_2d_pca,
        )

        # Load Global S2 distances to LUT
        global_data = pd.read_csv(global_dist_file)
        s2_to_lut_distances = global_data["distance_to_lut"].values

        # Compute S2 distances to RMs using same approach as panel A
        data = load_phase1_data(DATA_DIR)
        pca_results = perform_2d_pca(data)
        s2_pca = pca_results[TRAIT]["s2_pca"]
        rms_pca = pca_results[TRAIT]["rms_pca"]
        lut_pca = pca_results[TRAIT]["lut_pca"]

        # Use equal-sized subsamples of LUT and RMs for fair comparison
        min_ref_size = min(len(lut_pca), len(rms_pca))
        np.random.seed(42)

        # Subsample reference datasets to equal size
        lut_indices = np.random.choice(len(lut_pca), size=min_ref_size, replace=False)
        rms_indices = np.random.choice(len(rms_pca), size=min_ref_size, replace=False)

        lut_subset = lut_pca[lut_indices]
        rms_subset = rms_pca[rms_indices]

        # Calculate avg manhattan distance to 5 nearest neighbours
        nbrs_lut = NearestNeighbors(n_neighbors=5, metric="cityblock").fit(lut_subset)
        nbrs_rms = NearestNeighbors(n_neighbors=5, metric="cityblock").fit(rms_subset)

        # Calculate distances to nearest neighbors in the reference datasets
        dst_s2_to_lut, _ = nbrs_lut.kneighbors(s2_pca)
        dst_s2_to_rms, _ = nbrs_rms.kneighbors(s2_pca)

        # Also calculate RMs to LUT distances
        dst_rms_to_lut, _ = nbrs_lut.kneighbors(rms_subset)

        s2_to_lut_subset = dst_s2_to_lut.mean(axis=1)
        s2_to_rms_subset = dst_s2_to_rms.mean(axis=1)
        rms_to_lut_subset = dst_rms_to_lut.mean(axis=1)

        # Calculate statistics
        stats = {
            "Comparison": [
                "S2 → PROSAIL LUT",
                "S2 → GROUNDED-EO RMs",
                "GROUNDED-EO RMs → PROSAIL LUT",
            ],
            "Sample Size": [
                len(s2_to_lut_subset),
                len(s2_to_rms_subset),
                len(rms_to_lut_subset),
            ],
            "Mean Distance": [
                np.mean(s2_to_lut_subset),
                np.mean(s2_to_rms_subset),
                np.mean(rms_to_lut_subset),
            ],
            "Std Distance": [
                np.std(s2_to_lut_subset),
                np.std(s2_to_rms_subset),
                np.std(rms_to_lut_subset),
            ],
            "Median Distance": [
                np.median(s2_to_lut_subset),
                np.median(s2_to_rms_subset),
                np.median(rms_to_lut_subset),
            ],
            "Max Distance": [
                np.max(s2_to_lut_subset),
                np.max(s2_to_rms_subset),
                np.max(rms_to_lut_subset),
            ],
            "% > Cutoff (3.5)": [
                100 * np.mean(s2_to_lut_subset > 3.5),
                100 * np.mean(s2_to_rms_subset > 3.5),
                100 * np.mean(rms_to_lut_subset > 3.5),
            ],
        }

        # Create table
        table_data = []
        for i in range(len(stats["Comparison"])):
            row = [
                stats["Comparison"][i],
                f"{stats['Mean Distance'][i]:.3f}",
                f"{stats['% > Cutoff (3.5)'][i]:.1f}%",
            ]
            table_data.append(row)

        columns = ["Comparison", "Mean", "% > 3.5"]

        # Create table visualization
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=table_data, colLabels=columns, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Smaller font for inset
        table.scale(1, 1.8)  # Reduced scaling for inset

        # Make comparison column wider
        table.auto_set_column_width(col=[0, 1, 2])
        for i in range(len(table_data) + 1):  # +1 for header
            table[(i, 0)].set_width(0.5)  # Comparison column wider
            table[(i, 1)].set_width(0.2)  # Mean column
            table[(i, 2)].set_width(0.3)  # % > 3.5 column

        # Color code rows
        colors = ["blue", "red", "orange"]
        for i in range(len(table_data)):
            for j in range(len(columns)):
                table[(i + 1, j)].set_facecolor(colors[i])
                table[(i + 1, j)].set_alpha(0.7)

        # No title for inset table

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            style="italic",
            color="red",
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build data
    df_all = build_df_all()

    # Create 3x2 layout with bottom panel spanning both columns
    fig = plt.figure(figsize=(12, 12))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(3, 2, height_ratios=[1, 1, 1.1], figure=fig)

    # Top row: Performance and uncertainty
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Middle row: PCA and CDF comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Bottom row: Global map spanning both columns
    ax5 = fig.add_subplot(gs[2, :])

    # Top row: Performance and uncertainty
    # ax_a = plt.subplot(3, 2, 1)
    plot_performance_panel(ax1, df_all)
    add_panel_label(ax1, "A")

    # Top row: Performance and uncertainty
    plot_uncertainty_panel(ax2, df_all)
    add_panel_label(ax2, "B")

    # Middle row: PCA and CDF comparison
    plot_pca_panel(ax3)
    add_panel_label(ax3, "C")

    plot_distance_cdf_comparison(ax4)
    add_panel_label(ax4, "D")

    # Add inset for statistics table in bottom right of panel D
    ax_d_inset = inset_axes(
        ax4,
        width="70%",
        height="80%",
        loc="lower right",
        bbox_to_anchor=(
            0.20,
            -0.16,
            0.7,
            0.8,
        ),  # (left, bottom, width, height) inside plot
        bbox_transform=ax4.transAxes,
    )
    plot_distance_statistics_table(ax_d_inset)

    # Bottom row: Global map spanning both columns
    plot_global_map_placeholder_panel(ax5)
    add_panel_label(ax5, "E")

    plt.tight_layout()
    out_path = FIGURES_DIR / f"combined_domain_figure_{TRAIT}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined {TRAIT.upper()} figure: {out_path}")

    # # Create additional distance comparison figure
    # print(f"\nCreating additional distance comparison figure...")
    # create_distance_comparison_figure()


if __name__ == "__main__":
    main()
