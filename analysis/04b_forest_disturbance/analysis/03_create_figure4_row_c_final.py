"""Create the final Figure 4 row C with two fixed disturbance examples."""

from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plot_examples import fcover_path, read_on_template, read_pair_with_grid

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_DIR / "data"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGURE_DIR = ANALYSIS_DIR / "figures"
POLYGON_PATH = DATA_DIR / "sweden_completed_harvests_2020_2024.gpkg"
SUMMARY_PATH = RESULTS_DIR / "sweden_fcover_shift_summary_by_resolution_area.csv"

OUTPUT_STEM = FIGURE_DIR / "figure4_row_c_final"
DISTURBANCE_IDS = [
    "skogsstyrelsen_4586300",
    "skogsstyrelsen_5509601",
]
SITE_LABELS = ["Small site", "Large site"]
MAP_RESOLUTIONS = [20, 100, 300]
SUMMARY_RESOLUTIONS = [20, 100, 300, 500]
AREA_BIN_ORDER = ["<1 ha", "1–2 ha", "2–5 ha", ">5 ha"]
DIFFERENCE_LIMIT = 0.6
DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "difference_purple_white_green",
    ["purple", "white", "green"],
)


EPSG_EXAMPLES = "EPSG:32633"  # UTM zone 33N, Sweden
MGRS_EXAMPLES = ["33V", "33V"]  # MGRS tiles for Sweden


def adaptive_map_bounds(geometry) -> tuple[float, float, float, float]:
    """Zoom according to polygon dimensions while retaining minimal context."""

    minx, miny, maxx, maxy = geometry.bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    polygon_span = max(maxx - minx, maxy - miny)
    # Three polygon spans across each map, with a 500 m minimum map width.
    half_size = max(1.5 * polygon_span, 250)
    return (
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size,
    )


def add_scale_bar(axis, extent, distance_m: int = 300) -> None:
    """Add a high-contrast distance bar in projected map coordinates."""

    left, right, bottom, top = extent
    x_start = left + 0.08 * (right - left)
    y = bottom + 0.09 * (top - bottom)
    line = axis.plot(
        [x_start, x_start + distance_m],
        [y, y],
        color="black",
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=5,
    )[0]
    line.set_path_effects(
        [path_effects.Stroke(linewidth=3.8, foreground="white"), path_effects.Normal()]
    )
    label = axis.text(
        x_start + distance_m / 2,
        y + 0.025 * (top - bottom),
        f"{distance_m} m",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        zorder=5,
    )
    label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])


def plot_difference_map(
    axis,
    polygon,
    resolution_m: int,
    bounds,
    epsg_code: str,
    tile: str,
    add_ruler=False,
):
    year = int(polygon["disturbance_year"])

    # Read 20 m only to define the exact display grid
    _, _, template = read_pair_with_grid(
        fcover_path(year - 1, 20, tile, epsg_code),
        fcover_path(year + 1, 20, tile, epsg_code),
        bounds,
    )

    pre = read_on_template(
        fcover_path(year - 1, resolution_m, tile, epsg_code),
        template_shape=template["shape"],
        template_transform=template["transform"],
    )

    post = read_on_template(
        fcover_path(year + 1, resolution_m, tile, epsg_code),
        template_shape=template["shape"],
        template_transform=template["transform"],
    )

    difference = post - pre

    image = axis.imshow(
        difference.squeeze(),
        extent=template["extent"],
        origin="upper",
        cmap=DIFFERENCE_CMAP,
        vmin=-DIFFERENCE_LIMIT,
        vmax=DIFFERENCE_LIMIT,
        interpolation="nearest",
    )

    gpd.GeoSeries([polygon.geometry]).boundary.plot(
        ax=axis,
        color="black",
        linewidth=1.15,
    )

    if add_ruler:
        add_scale_bar(
            axis,
            template["extent"],
            distance_m=100 if polygon["area_ha"] < 1 else 300,
        )

    axis.set_aspect("equal")
    axis.set_axis_off()

    return image


def plot_summary(axis, summary: pd.DataFrame) -> None:
    """Draw the complete-case aggregate summary with publication-scale text."""

    colors = sns.color_palette("viridis", n_colors=len(AREA_BIN_ORDER))
    x_positions = np.arange(len(SUMMARY_RESOLUTIONS))
    x_offsets = np.linspace(-0.12, 0.12, len(AREA_BIN_ORDER))

    for area_bin, color, offset in zip(AREA_BIN_ORDER, colors, x_offsets):
        values = (
            summary[summary["area_bin"] == area_bin]
            .set_index("resolution_m")
            .reindex(SUMMARY_RESOLUTIONS)
        )
        median = values["median"].to_numpy(dtype=float)
        lower = median - values["q25"].to_numpy(dtype=float)
        upper = values["q75"].to_numpy(dtype=float) - median
        sample_size = int(values["n"].dropna().iloc[0])
        axis.errorbar(
            x_positions + offset,
            median,
            yerr=np.vstack([lower, upper]),
            label=f"{area_bin} (n={sample_size:,})",
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.6,
            elinewidth=1.1,
            capsize=2.5,
            capthick=1.1,
        )
    axis.axhline(0, color="0.25", linestyle="--", linewidth=0.9, zorder=0)
    axis.set_xticks(x_positions, SUMMARY_RESOLUTIONS)
    axis.set_ylim(-0.5, 0.04)
    axis.set_xlabel("Resolution (m)", fontsize=11)
    axis.set_ylabel("")
    axis.set_title(
        "Δ FCOVER (post − pre)",
        loc="left",
        fontsize=10.5,
        pad=5,
    )
    axis.tick_params(labelsize=9.5, length=3.5, width=0.9)
    axis.legend(
        title="Harvest area",
        fontsize=9.8,
        title_fontsize=11.3,
        frameon=False,
        loc="lower right",
        handlelength=1.5,
        labelspacing=0.45,
        borderaxespad=2.50,
    )
    for spine in ("left", "bottom"):
        axis.spines[spine].set_linewidth(1.1)
    sns.despine(ax=axis)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    polygons = gpd.read_file(POLYGON_PATH)
    polygons = polygons[polygons["disturbance_id"].isin(DISTURBANCE_IDS)]
    polygons = polygons.to_crs("EPSG:32633").set_index("disturbance_id")
    missing = set(DISTURBANCE_IDS) - set(polygons.index)

    if missing:
        raise ValueError(f"Missing requested disturbance polygons: {sorted(missing)}")

    sns.set_theme(
        style="ticks",
        context="paper",
    )

    # ------------------------------------------------------------------
    # Figure layout
    #
    # colorbar | site label | 3 example maps | aggregate summary
    # ------------------------------------------------------------------

    figure = plt.figure(
        figsize=(10, 4.0),
    )

    grid = figure.add_gridspec(
        2,
        7,
        width_ratios=[
            0.035,  # colorbar
            0.08,  # vertical site info
            0.5,  # 20 m
            0.5,  # 100 m
            0.5,  # 300 m
            0.1,  # spacer
            1.15,  # aggregate summary
        ],
        wspace=0.00,
        hspace=0.01,
    )

    colorbar_axis = figure.add_subplot(grid[:, 0])

    label_axes = [figure.add_subplot(grid[row, 1]) for row in range(2)]

    map_axes = [
        [figure.add_subplot(grid[row, column]) for column in range(2, 5)]
        for row in range(2)
    ]

    # intentionally unused:
    # grid[:, 5]

    summary_axis = figure.add_subplot(grid[:, 6])

    # ------------------------------------------------------------------
    # Example sites
    # ------------------------------------------------------------------

    map_image = None

    for row, (disturbance_id, site_label) in enumerate(
        zip(DISTURBANCE_IDS, SITE_LABELS)
    ):
        epsg_code = EPSG_EXAMPLES
        tile = MGRS_EXAMPLES[row]

        polygon = polygons.loc[disturbance_id]
        bounds = adaptive_map_bounds(polygon.geometry)

        # --------------------------------------------------------------
        # Minimal vertical site information
        # --------------------------------------------------------------

        label_axes[row].text(
            0.8,
            0.5,
            (f"{polygon['area_ha']:.1f} ha"),
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
        )

        label_axes[row].set_axis_off()

        # --------------------------------------------------------------
        # Resolution maps
        # --------------------------------------------------------------

        for column, resolution_m in enumerate(MAP_RESOLUTIONS):
            axis = map_axes[row][column]

            if row == 0:
                axis.set_title(
                    f"{resolution_m} m",
                    fontsize=10,
                    pad=4,
                )

            map_image = plot_difference_map(
                axis,
                polygon,
                resolution_m,
                bounds,
                epsg_code=epsg_code,
                tile=tile,
                add_ruler=(column == 0),
            )

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------

    colorbar = figure.colorbar(
        map_image,
        cax=colorbar_axis,
        orientation="vertical",
    )

    colorbar.ax.set_title(
        "Δ FCOVER",
        fontsize=11.0,
        pad=6,
    )

    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.tick_params(
        axis="y",
        labelleft=True,
        labelright=False,
        left=True,
        right=False,
    )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------

    plot_summary(
        summary_axis,
        summary,
    )

    # ------------------------------------------------------------------
    # Panel label
    # ------------------------------------------------------------------

    # figure.text(
    #     0.008,
    #     0.975,
    #     "C",
    #     fontsize=13,
    #     fontweight="bold",
    #     va="top",
    # )

    figure.subplots_adjust(
        left=0.025,
        right=0.995,
        top=0.91,
        bottom=0.11,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    for suffix, options in {
        ".png": {"dpi": 500},
        ".pdf": {},
        ".svg": {},
    }.items():
        output_path = OUTPUT_STEM.with_suffix(suffix)

        figure.savefig(
            output_path,
            bbox_inches="tight",
            **options,
        )

        print(f"Saved {output_path}")

    plt.close(figure)


if __name__ == "__main__":
    main()
