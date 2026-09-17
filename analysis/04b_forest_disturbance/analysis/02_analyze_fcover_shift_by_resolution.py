"""Analyse pre/post-harvest FCOVER shifts across spatial resolutions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_DIR / "data"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURE_DIR = ANALYSIS_DIR / "figures"

INPUT_PATH = RESULTS_DIR / "sweden_fcover_pre_post_intervention_zonal_means.csv"
COMPLETE_DATA_PATH = RESULTS_DIR / "sweden_fcover_complete_resolution_shifts.csv"
SUMMARY_PATH = RESULTS_DIR / "sweden_fcover_shift_summary_by_resolution_area.csv"
FIGURE_PATH = FIGURE_DIR / "fcover_shift_by_resolution_and_area.png"
FIGURE_PDF_PATH = FIGURE_DIR / "fcover_shift_by_resolution_and_area.pdf"
COMPACT_FIGURE_PATH = FIGURE_DIR / "figure_3c_fcover_shift_compact.png"
COMPACT_FIGURE_PDF_PATH = FIGURE_DIR / "figure_3c_fcover_shift_compact.pdf"
COMPACT_FIGURE_SVG_PATH = FIGURE_DIR / "figure_3c_fcover_shift_compact.svg"

RESOLUTIONS = [20, 100, 300, 500]
AREA_BIN_ORDER = ["<1 ha", "1–2 ha", "2–5 ha", ">5 ha"]
ID_COLUMNS = [
    "disturbance_id",
    "area_ha",
    "disturbance_type",
    "disturbance_year",
]


def assign_area_bins(area_ha: pd.Series) -> pd.Categorical:
    """Assign the requested area classes without ambiguous boundary values."""

    labels = np.select(
        [
            area_ha < 1,
            (area_ha >= 1) & (area_ha < 2),
            (area_ha >= 2) & (area_ha <= 5),
            area_ha > 5,
        ],
        AREA_BIN_ORDER,
        default=None,
    )
    return pd.Categorical(labels, categories=AREA_BIN_ORDER, ordered=True)


def prepare_complete_shifts(input_path: Path) -> pd.DataFrame:
    """Pair pre/post values and retain polygons complete at every resolution."""

    columns = [*ID_COLUMNS, "intervention_period", "resolution_m", "fcover_mean"]
    data = pd.read_csv(input_path, usecols=columns)
    data = data[
        data["resolution_m"].isin(RESOLUTIONS)
        & data["intervention_period"].isin(["pre", "post"])
    ].copy()

    # drop missing
    data = data.dropna(subset=["fcover_mean"])

    # drop all disturbance _ids that have more than 1 row for any resolution/period combination

    counts = data.groupby(
        ["disturbance_id", "resolution_m", "intervention_period"], observed=True
    ).size()
    complete_ids = counts[counts == 1].index
    data = data[
        data.set_index(
            ["disturbance_id", "resolution_m", "intervention_period"]
        ).index.isin(complete_ids)
    ].copy()

    observation_key = [*ID_COLUMNS, "resolution_m", "intervention_period"]
    duplicates = data.duplicated(observation_key, keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate polygon/resolution/period rows; "
            "pre/post pairing would be ambiguous"
        )

    paired = (
        data.pivot(
            index=[*ID_COLUMNS, "resolution_m"],
            columns="intervention_period",
            values="fcover_mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    paired = paired.dropna(subset=["pre", "post"])

    polygon_key = ["disturbance_id", "disturbance_year"]
    resolution_counts = paired.groupby(polygon_key, observed=True)[
        "resolution_m"
    ].nunique()
    complete_keys = resolution_counts[resolution_counts == len(RESOLUTIONS)].index
    complete_index = pd.MultiIndex.from_frame(paired[polygon_key])
    paired = paired[complete_index.isin(complete_keys)].copy()

    observed_resolutions = paired.groupby(polygon_key, observed=True)[
        "resolution_m"
    ].agg(set)
    expected_resolutions = set(RESOLUTIONS)
    if not observed_resolutions.map(
        lambda values: values == expected_resolutions
    ).all():
        raise RuntimeError("Completeness filtering failed for one or more polygons")

    # Positive values represent FCOVER loss following harvest.
    paired["fcover_difference_post_minus_pre"] = paired["post"] - paired["pre"]
    paired["area_bin"] = assign_area_bins(paired["area_ha"])
    paired = paired.dropna(subset=["area_bin"])
    paired["resolution_m"] = pd.Categorical(
        paired["resolution_m"],
        categories=RESOLUTIONS,
        ordered=True,
    )
    return paired.sort_values([*polygon_key, "resolution_m"]).reset_index(drop=True)


def summarise_shifts(complete: pd.DataFrame) -> pd.DataFrame:
    """Summarise FCOVER shifts for every resolution and area-bin combination."""

    value = "fcover_difference_post_minus_pre"
    return (
        complete.groupby(["resolution_m", "area_bin"], observed=False)[value]
        .agg(
            n="count",
            mean="mean",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def plot_shifts(complete: pd.DataFrame) -> None:
    """Create grouped boxplots with equally spaced resolution categories."""

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    sns.boxplot(
        data=complete,
        x="resolution_m",
        y="fcover_difference_post_minus_pre",
        hue="area_bin",
        order=RESOLUTIONS,
        hue_order=AREA_BIN_ORDER,
        palette="viridis",
        showfliers=False,
        width=0.78,
        linewidth=1,
        ax=axis,
    )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="FCOVER resolution (m)",
        ylabel="FCOVER difference (post − pre harvest)",
        title="Detectable FCOVER shift by resolution and harvest area",
    )
    axis.legend(title="Harvest area", frameon=True, ncol=2)
    sns.despine(ax=axis)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    figure.savefig(FIGURE_PDF_PATH, bbox_inches="tight")
    plt.close(figure)


def plot_compact_shifts(summary: pd.DataFrame) -> None:
    """Create a small point-range panel suitable for a multipanel figure."""

    sns.set_theme(style="ticks", context="paper")
    figure, axis = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    colors = sns.color_palette("viridis", n_colors=len(AREA_BIN_ORDER))
    x_positions = np.arange(len(RESOLUTIONS))
    x_offsets = np.linspace(-0.12, 0.12, len(AREA_BIN_ORDER))

    for area_bin, color, x_offset in zip(AREA_BIN_ORDER, colors, x_offsets):
        values = (
            summary[summary["area_bin"] == area_bin]
            .set_index("resolution_m")
            .reindex(RESOLUTIONS)
        )
        median = values["median"].to_numpy(dtype=float)
        lower = median - values["q25"].to_numpy(dtype=float)
        upper = values["q75"].to_numpy(dtype=float) - median
        sample_sizes = values["n"].dropna().astype(int).unique()
        if len(sample_sizes) != 1:
            raise RuntimeError(
                f"Sample size for {area_bin} differs among resolutions: "
                f"{sample_sizes.tolist()}"
            )
        legend_label = f"{area_bin} (n={sample_sizes[0]:,})"
        axis.errorbar(
            x_positions + x_offset,
            median,
            yerr=np.vstack([lower, upper]),
            label=legend_label,
            color=color,
            marker="o",
            markersize=3.8,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
        )

    axis.axhline(0, color="0.25", linestyle="--", linewidth=0.7, zorder=0)
    axis.set_xticks(x_positions, RESOLUTIONS)
    axis.set_xlabel("Resolution (m)")
    axis.set_ylabel("FCOVER change\n(post − pre harvest)")
    axis.tick_params(labelsize=8, length=3)
    axis.legend(
        title="Harvest area",
        fontsize=7,
        title_fontsize=7,
        frameon=False,
        ncol=1,
        loc="lower right",
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    axis.text(
        -0.16,
        1.04,
        "C",
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    sns.despine(ax=axis)

    figure.savefig(COMPACT_FIGURE_PATH, dpi=600, bbox_inches="tight")
    figure.savefig(COMPACT_FIGURE_PDF_PATH, bbox_inches="tight")
    figure.savefig(COMPACT_FIGURE_SVG_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing zonal extraction table: {INPUT_PATH}. Run explore_data.py first."
        )

    complete = prepare_complete_shifts(INPUT_PATH)
    summary = summarise_shifts(complete)
    polygon_count = complete[["disturbance_id", "disturbance_year"]].drop_duplicates()

    complete.to_csv(COMPLETE_DATA_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_shifts(complete)
    plot_compact_shifts(summary)

    print(f"Complete polygons: {len(polygon_count)}")
    print(f"Complete polygon-resolution rows: {len(complete)}")
    print(f"Saved complete data: {COMPLETE_DATA_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved figure: {FIGURE_PDF_PATH}")
    print(f"Saved compact figure: {COMPACT_FIGURE_PATH}")
    print(f"Saved compact figure: {COMPACT_FIGURE_PDF_PATH}")
    print(f"Saved compact figure: {COMPACT_FIGURE_SVG_PATH}")


if __name__ == "__main__":
    main()
