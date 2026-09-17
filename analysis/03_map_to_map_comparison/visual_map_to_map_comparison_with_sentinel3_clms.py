"""Nine-panel map comparison including temporally matched Sentinel-3 CLMS."""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from compare_s2biophys_and_sentinel3_clms import (
    CACHE_DIR,
    FIGURE_DIR,
    REGIONS_FIGURE,
    align_at_clms_resolution,
    load_clms_temporal_mean,
)
from helpers import Region, run_region_comparison
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

TRAITS = {
    "fapar": {
        "variables": {
            "s2biophys": "fapar_s2biophys_mean",
            "sl2p": "fapar_sl2p_mean",
            "groundedeo": "fapar_groundedeo_mean",
        },
        "map_labels": {
            "s2biophys": "S2BIOPHYS",
            "sl2p": "SL2P",
            "groundedeo": "GROUNDED-EO GPR",
        },
        "axis_labels": {
            "s2biophys": "S2BIOPHYS",
            "sl2p": "SL2P",
            "groundedeo": "GROUNDED-EO GPR",
            "clms": "Sentinel-3 CLMS",
        },
        "value_label": "FAPAR",
        "limits": (0.0, 1.0),
        "cmap": LinearSegmentedColormap.from_list(
            "combined_fapar",
            ["#ffffdd", "#e6ad12", "#c53859", "#3a26a1", "#000000"],
        ),
    },
    "lai": {
        "variables": {
            "s2biophys": "laie_s2biophys_mean",
            "sl2p": "laie_sl2p_mean",
            "groundedeo": "lai_groundedeo_mean",
        },
        "map_labels": {
            "s2biophys": "S2BIOPHYS (LAIe)",
            "sl2p": "SL2P (LAIe)",
            "groundedeo": "GROUNDED-EO GPR (LAI)",
        },
        "axis_labels": {
            "s2biophys": "S2BIOPHYS",
            "sl2p": "SL2P",
            "groundedeo": "GROUNDED-EO GPR",
            "clms": "Sentinel-3 CLMS",
        },
        "value_label": "LAIe / LAI",
        "limits": (0.0, 6.0),
        "cmap": LinearSegmentedColormap.from_list(
            "combined_lai",
            [
                "#fffdcd",
                "#e1cd73",
                "#aaac20",
                "#5f920c",
                "#187328",
                "#144b2a",
                "#172313",
            ],
        ),
    },
}


def _extent(da: xr.DataArray | xr.Dataset) -> list[float]:
    return [
        float(da.x.min()),
        float(da.x.max()),
        float(da.y.min()),
        float(da.y.max()),
    ]


def _plot_relationship(
    ax,
    x_da: xr.DataArray,
    y_da: xr.DataArray,
    *,
    limits: tuple[float, float],
    x_label: str,
    y_label: str,
) -> None:
    """Draw a log-density pixel relationship plot and summary statistics."""
    x = x_da.values.ravel()
    y = y_da.values.ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size:
        ax.hexbin(
            x,
            y,
            gridsize=35,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        rho = spearmanr(x, y).statistic
        metrics = (
            rf"$r_s$ = {rho:.2f}"
            "\n"
            rf"median $\Delta$ = {np.median(x - y):.2f}"
            "\n"
            f"n = {x.size}"
        )
    else:
        metrics = "No overlapping valid pixels"

    vmin, vmax = limits
    if x.size:
        # Keep the standard comparison range, but do not clip observations
        # when either product contains values above its map-display maximum.
        vmax = max(vmax, float(np.nanmax(x)), float(np.nanmax(y)))
    ax.plot(
        [vmin, vmax],
        [vmin, vmax],
        linestyle="--",
        linewidth=0.8,
        color="black",
    )
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label, fontsize=6.5)
    ax.set_ylabel(y_label, fontsize=6.5)
    ax.tick_params(labelsize=6.5, length=2)
    ax.text(
        0.05,
        0.95,
        metrics,
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
        bbox={
            "facecolor": "white",
            "alpha": 0.78,
            "edgecolor": "none",
            "pad": 1.5,
        },
    )


def prepare_region(
    region: Region,
    *,
    trait: Literal["fapar", "lai"],
    gee_project: str,
    use_cache: bool = True,
) -> dict:
    """Load one region and prepare the matched 20 m and 300 m products."""
    retrievals, s2_median = run_region_comparison(
        region,
        gee_project=gee_project,
        use_cache=use_cache,
        cache_dir=CACHE_DIR,
    )
    config = TRAITS[trait]
    s2biophys_20m = retrievals[config["variables"]["s2biophys"]]
    clms_temporal_mean = load_clms_temporal_mean(
        region,
        trait,
        use_cache=use_cache,
    )
    s2biophys_300m, clms_300m = align_at_clms_resolution(
        s2biophys_20m,
        clms_temporal_mean,
        utm_epsg=int(retrievals.attrs["utm_epsg"]),
    )
    return {
        "region": region,
        "retrievals": retrievals,
        "s2_median": s2_median,
        "s2biophys_300m": s2biophys_300m,
        "clms_300m": clms_300m,
        "clms_dekad_count": int(clms_temporal_mean.attrs["dekad_count"]),
    }


def plot_combined_comparison(
    rows: list[dict],
    *,
    trait: Literal["fapar", "lai"],
    rgb_max: float = 0.3,
):
    """Plot the nine-column S2 retrieval and Sentinel-3 comparison figure."""
    if not rows:
        raise ValueError("At least one regional comparison is required.")

    config = TRAITS[trait]
    variables = config["variables"]
    vmin, vmax = config["limits"]
    n_regions = len(rows)

    # Columns: RGB, three retrieval maps, two 20 m relationships,
    # S2BIOPHYS at 300 m, CLMS at 300 m, and their relationship.
    fig = plt.figure(
        figsize=(18, 2.25 * n_regions + 0.9),
        constrained_layout=False,
    )
    # The three narrow columns are reserved for relationship y-axis labels.
    # All actual panels retain equal widths, while map-to-map gaps stay tight.
    grid = fig.add_gridspec(
        n_regions,
        13,
        width_ratios=[1, 1, 1, 1, 0.15, 1, 0.15, 1, 0.22, 1, 1, 0.15, 1],
        wspace=0.06,
        hspace=0.07,
    )
    panel_columns = [0, 1, 2, 3, 5, 7, 9, 10, 12]
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row, column]) for column in panel_columns]
            for row in range(n_regions)
        ]
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.995,
        top=0.89,
        bottom=0.115,
    )

    map_image = None
    for row_index, row in enumerate(rows):
        row_axes = axes[row_index]
        retrievals = row["retrievals"]
        s2_median = row["s2_median"]
        maps = {name: retrievals[var] for name, var in variables.items()}

        rgb = np.stack(
            [s2_median["B4"], s2_median["B3"], s2_median["B2"]],
            axis=-1,
        ).astype(float)
        rgb = np.clip(rgb / rgb_max, 0, 1)
        extent_20m = _extent(retrievals)
        extent_300m = _extent(row["clms_300m"])

        row_axes[0].imshow(rgb, extent=extent_20m, origin="upper")
        for ax, product in zip(row_axes[1:4], maps.values()):
            map_image = ax.imshow(
                product,
                extent=extent_20m,
                origin="upper",
                cmap=config["cmap"],
                vmin=vmin,
                vmax=vmax,
            )

        _plot_relationship(
            row_axes[4],
            maps["s2biophys"],
            maps["sl2p"],
            limits=config["limits"],
            x_label=config["axis_labels"]["s2biophys"],
            y_label=config["axis_labels"]["sl2p"],
        )
        _plot_relationship(
            row_axes[5],
            maps["s2biophys"],
            maps["groundedeo"],
            limits=config["limits"],
            x_label=config["axis_labels"]["s2biophys"],
            y_label=config["axis_labels"]["groundedeo"],
        )

        for ax, product in zip(
            row_axes[6:8],
            [row["s2biophys_300m"], row["clms_300m"]],
        ):
            map_image = ax.imshow(
                product,
                extent=extent_300m,
                origin="upper",
                cmap=config["cmap"],
                vmin=vmin,
                vmax=vmax,
            )

        _plot_relationship(
            row_axes[8],
            row["s2biophys_300m"],
            row["clms_300m"],
            limits=config["limits"],
            x_label="S2BIOPHYS 300 m",
            y_label=config["axis_labels"]["clms"],
        )

        for map_ax in [*row_axes[:4], *row_axes[6:8]]:
            map_ax.set_xticks([])
            map_ax.set_yticks([])
            map_ax.set_aspect("equal")

        row_axes[0].set_ylabel(
            row["region"].ecosystem,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            labelpad=11,
        )

        if row_index < n_regions - 1:
            for relationship_ax in [row_axes[4], row_axes[5], row_axes[8]]:
                relationship_ax.set_xlabel("")
                relationship_ax.tick_params(axis="x", labelbottom=False)

    column_titles = [
        "Sentinel-2 RGB",
        config["map_labels"]["s2biophys"],
        config["map_labels"]["sl2p"],
        config["map_labels"]["groundedeo"],
        "S2BIOPHYS vs SL2P\n(20 m pixels)",
        "S2BIOPHYS vs GROUNDED-EO\n(20 m pixels)",
        "S2BIOPHYS 300 m\n(mean aggregation)",
        "Sentinel-3 CLMS 300 m\n(mean of 10-day products)",
        "S2BIOPHYS vs CLMS\n(300 m pixels)",
    ]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=8.5, pad=5)

    # Separate the native-resolution benchmark panels from the 300 m
    # cross-sensor comparison with one unobtrusive rule.
    fig.canvas.draw()
    divider_x = (axes[0, 5].get_position().x1 + axes[0, 6].get_position().x0) / 2
    fig.add_artist(
        Line2D(
            [divider_x, divider_x],
            [axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1],
            transform=fig.transFigure,
            color="#B0B0B0",
            linewidth=0.8,
        )
    )
    fig.text(
        axes[0, 0].get_position().x0,
        axes[0, 0].get_position().y1 + 0.045,
        "A",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        axes[0, 6].get_position().x0,
        axes[0, 6].get_position().y1 + 0.045,
        "B",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    if trait == "lai":
        fig.suptitle(
            "S2BIOPHYS and SL2P represent effective LAI (LAIe); "
            "GROUNDED-EO and CLMS represent LAI",
            fontsize=9,
            y=1.005,
        )

    # Keep map colorbars clear of the three relationship x labels.
    fig.canvas.draw()
    colorbar_left = axes[-1, 1].get_position().x0
    colorbar_right = axes[-1, 3].get_position().x1
    colorbar_bottom = axes[-1, 1].get_position().y0 - 0.045
    colorbar_ax = fig.add_axes(
        [colorbar_left, colorbar_bottom, colorbar_right - colorbar_left, 0.012]
    )
    colorbar = fig.colorbar(
        map_image,
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label(config["value_label"], fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    colorbar_300m_left = axes[-1, 6].get_position().x0
    colorbar_300m_right = axes[-1, 7].get_position().x1
    colorbar_300m_ax = fig.add_axes(
        [
            colorbar_300m_left,
            colorbar_bottom,
            colorbar_300m_right - colorbar_300m_left,
            0.012,
        ]
    )
    colorbar_300m = fig.colorbar(
        map_image,
        cax=colorbar_300m_ax,
        orientation="horizontal",
    )
    colorbar_300m.set_label(config["value_label"], fontsize=8)
    colorbar_300m.ax.tick_params(labelsize=7)
    return fig, axes


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trait in ("fapar", "lai"):
            rows = []
            for region in REGIONS_FIGURE:
                print(f"Preparing {trait.upper()}: {region.name}")
                rows.append(
                    prepare_region(
                        region,
                        trait=trait,
                        gee_project="ee-speckerfelix",
                        use_cache=True,
                    )
                )

            fig, _ = plot_combined_comparison(rows, trait=trait)
            output = FIGURE_DIR / (
                f"visual_map_to_map_comparison_with_sentinel3_clms_{trait}.png"
            )
            fig.savefig(output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
