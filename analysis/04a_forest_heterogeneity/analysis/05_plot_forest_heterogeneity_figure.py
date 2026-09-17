#!/usr/bin/env python3
"""Create the complete matched-forest LAIe heterogeneity figure (panels A-D)."""

from __future__ import annotations

import argparse
import os
from functools import cache
from io import BytesIO
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image
from rasterio.plot import plotting_extent
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

PLANTATION_COLOR = "#3569B7"
CONTROL_COLOR = "#D95F02"
LAIE_COLORS = [
    "#fffdcd",
    "#e1cd73",
    "#aaac20",
    "#5f920c",
    "#187328",
    "#144b2a",
    "#172313",
]
LAIE_CMAP = mcolors.LinearSegmentedColormap.from_list("lai", LAIE_COLORS, N=256)
LAIE_NORM = mcolors.Normalize(vmin=0, vmax=5.0, clip=True)
METRICS = {
    "mean": "Mean LAIe",
    "sd": "LAIe SD",
    # "rao_q": "Rao's Q",
    "adjacent_difference": "Adjacent contrast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laie", type=Path, required=True)
    parser.add_argument(
        "--jrc",
        type=Path,
        required=True,
        help="JRC Global Forest Types raster aligned to the LAIe grid.",
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="CSV from analyse_matched_pair_laie_heterogeneity.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-layer", default="matched_pairs")
    parser.add_argument(
        "--pair-id",
        type=int,
        default=380,
        help="Example pair; default is closest to median SD difference",
    )
    parser.add_argument("--window-pixels", type=int, default=25)
    parser.add_argument("--padding-m", type=float, default=300)
    parser.add_argument("--laie-scale-factor", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def centered_window(row: int, col: int, size: int) -> Window:
    half = size // 2
    return Window(col - half, row - half, size, size)


def read_laie_window(src, row: int, col: int, size: int, scale: float) -> np.ndarray:
    data = src.read(1, window=centered_window(row, col, size), masked=True)
    return np.asarray(data.filled(np.nan), dtype=float) * scale


def stretch_rgb(data: np.ma.MaskedArray) -> np.ndarray:
    image = np.moveaxis(data.filled(np.nan).astype(float), 0, -1)
    output = np.zeros_like(image)
    for band in range(3):
        values = image[..., band]
        valid = np.isfinite(values)
        if not valid.any():
            continue
        low, high = np.nanpercentile(values, (2, 98))
        output[..., band] = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    output[~np.isfinite(image).all(axis=2)] = 1
    return output


def bootstrap_mean(values: np.ndarray, samples: int, rng) -> tuple[float, float, float]:
    """Return the mean paired difference and its percentile bootstrap interval."""
    observed = values.mean()
    estimates = np.empty(samples)
    chunk = 2_000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        boot = values[rng.integers(0, len(values), size=(stop - start, len(values)))]
        estimates[start:stop] = boot.mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(observed), float(lower), float(upper)


def add_window_boxes(ax, pair, transform, size: int) -> None:
    for prefix, color, short_label in (
        ("plant", PLANTATION_COLOR, "P"),
        ("control", CONTROL_COLOR, "F"),
    ):
        window = centered_window(
            int(getattr(pair, f"{prefix}_row")),
            int(getattr(pair, f"{prefix}_col")),
            size,
        )
        left, bottom, right, top = window_bounds(window, transform)
        ax.add_patch(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
            )
        )
        ax.text(
            left,
            top,
            short_label,
            color="white",
            fontsize=7,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
        )


def add_scale_bar(ax, length_m: float = 200) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = xmin + 0.06 * (xmax - xmin)
    y = ymin + 0.07 * (ymax - ymin)
    ax.plot(
        [x, x + length_m], [y, y], color="white", linewidth=3, solid_capstyle="butt"
    )
    ax.text(
        x + length_m / 2,
        y + 0.025 * (ymax - ymin),
        f"{int(length_m)} m",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
        path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
    )


def add_jrc_forest_boundaries(
    ax, jrc_crop: np.ma.MaskedArray, extent: tuple[float, float, float, float]
) -> None:
    """Draw a white outline around JRC plantation pixels (class 20)."""
    values = jrc_crop.filled(0)
    plantation = values == 20
    rows, cols = values.shape
    left, right, bottom, top = extent
    pixel_width = (right - left) / cols
    pixel_height = (top - bottom) / rows
    segments = []

    # Interfaces between horizontally adjacent pixels (vertical line segments).
    vertical = plantation[:, :-1] != plantation[:, 1:]
    for row, col in np.argwhere(vertical):
        x = left + (col + 1) * pixel_width
        y_top = top - row * pixel_height
        segments.append(((x, y_top), (x, y_top - pixel_height)))

    # Interfaces between vertically adjacent pixels (horizontal line segments).
    horizontal = plantation[:-1, :] != plantation[1:, :]
    for row, col in np.argwhere(horizontal):
        x_left = left + col * pixel_width
        y = top - (row + 1) * pixel_height
        segments.append(((x_left, y), (x_left + pixel_width, y)))

    if segments:
        boundary = LineCollection(
            segments,
            colors="white",
            linewidths=1.3,
            linestyles="solid",
            zorder=3,
        )
        boundary.set_path_effects(
            [path_effects.withStroke(linewidth=2.2, foreground="#222222")]
        )
        ax.add_collection(boundary)


@cache
def get_sentinel2_rgb(
    geometry_bounds: tuple[float, float, float, float],
    geometry_crs: str = "EPSG:4326",
    *,
    start_date: str = "2020-01-01",
    end_date: str = "2021-01-01",
    dimensions: int = 1200,
    cloud_score_threshold: float = 0.65,
    reflectance_min: float = 0,
    reflectance_max: float = 2500,
    gamma: float = 1.1,
) -> np.ndarray:
    """
    Create and download a cloud-masked Sentinel-2 RGB median composite.

    Parameters
    ----------
    geometry
        Earth Engine geometry defining the requested area.
    start_date, end_date
        Half-open temporal interval [start_date, end_date).
    dimensions
        Maximum thumbnail dimension in pixels. The aspect ratio is retained.
    cloud_score_threshold
        Minimum Cloud Score+ ``cs_cdf`` value. Higher values retain only
        clearer pixels.
    reflectance_min, reflectance_max
        Visualization range for Sentinel-2 reflectance scaled by 10,000.
    gamma
        RGB visualization gamma.

    Returns
    -------
    np.ndarray
        RGB image with shape (height, width, 3) and dtype uint8.
    """

    ee.Initialize()
    geometry = ee.Geometry.Rectangle(geometry_bounds, proj=geometry_crs, geodesic=False)

    sentinel2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .linkCollection(
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"),
            ["cs_cdf"],
        )
    )

    def mask_image(image: ee.Image) -> ee.Image:
        # Cloud Score+ mask
        clear = image.select("cs_cdf").gte(cloud_score_threshold)

        # Additional Sentinel-2 Scene Classification Layer screening
        scl = image.select("SCL")
        valid_scl = (
            scl.neq(1)  # saturated/defective
            .And(scl.neq(3))  # cloud shadow
            .And(scl.neq(8))  # medium-probability cloud
            .And(scl.neq(9))  # high-probability cloud
            .And(scl.neq(10))  # cirrus
            .And(scl.neq(11))  # snow/ice
        )

        return image.updateMask(clear.And(valid_scl)).select(["B4", "B3", "B2"])

    rgb_composite = sentinel2.map(mask_image).median().clip(geometry)

    # Convert reflectance bands into a rendered 8-bit RGB image.
    visualized = rgb_composite.visualize(
        bands=["B4", "B3", "B2"],
        min=reflectance_min,
        max=reflectance_max,
        gamma=gamma,
    )

    thumbnail_url = visualized.getThumbURL(
        {
            "region": geometry,
            "dimensions": dimensions,
            "format": "png",
        }
    )

    response = requests.get(thumbnail_url, timeout=120)
    response.raise_for_status()

    return np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))


def main() -> None:
    args = parse_args()
    if args.window_pixels % 2 == 0:
        raise ValueError("window-pixels must be odd.")

    pairs = gpd.read_file(args.pairs, layer=args.pairs_layer)
    metrics = pd.read_csv(args.metrics)
    if "passes_valid_fraction" in metrics.columns:
        passed = metrics["passes_valid_fraction"]
        if passed.dtype != bool:
            passed = passed.astype(str).str.lower().eq("true")
        metrics = metrics.loc[passed].copy()

    if args.pair_id is None:
        median_sd = metrics["difference_sd"].median()
        example_id = int(
            metrics.loc[
                (metrics["difference_sd"] - median_sd).abs().idxmin(), "pair_id"
            ]
        )
    else:
        example_id = args.pair_id
    selected = pairs.loc[pairs["pair_id"] == example_id]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one pair with pair_id={example_id}.")
    pair = next(selected.itertuples(index=False))

    with rasterio.open(args.laie) as laie_src:
        if pairs.crs != laie_src.crs:
            raise ValueError("Pairs and LAIe raster must use the same CRS.")
        p_window = centered_window(
            int(pair.plant_row), int(pair.plant_col), args.window_pixels
        )
        c_window = centered_window(
            int(pair.control_row), int(pair.control_col), args.window_pixels
        )
        p_bounds = window_bounds(p_window, laie_src.transform)
        c_bounds = window_bounds(c_window, laie_src.transform)
        crop_bounds = (
            min(p_bounds[0], c_bounds[0]) - args.padding_m,
            min(p_bounds[1], c_bounds[1]) - args.padding_m,
            max(p_bounds[2], c_bounds[2]) + args.padding_m,
            max(p_bounds[3], c_bounds[3]) + args.padding_m,
        )
        laie_crop_window = from_bounds(*crop_bounds, transform=laie_src.transform)
        laie_crop = laie_src.read(1, window=laie_crop_window, masked=True).astype(float)
        laie_crop *= args.laie_scale_factor
        laie_extent = plotting_extent(
            laie_crop, laie_src.window_transform(laie_crop_window)
        )
        plantation_values = read_laie_window(
            laie_src,
            int(pair.plant_row),
            int(pair.plant_col),
            args.window_pixels,
            args.laie_scale_factor,
        ).ravel()
        control_values = read_laie_window(
            laie_src,
            int(pair.control_row),
            int(pair.control_col),
            args.window_pixels,
            args.laie_scale_factor,
        ).ravel()
        plantation_values = plantation_values[np.isfinite(plantation_values)]
        control_values = control_values[np.isfinite(control_values)]
        laie_transform = laie_src.transform
        laie_crs = laie_src.crs

    with rasterio.open(args.jrc) as jrc_src:
        if jrc_src.crs != laie_crs:
            raise ValueError("JRC and LAIe rasters must use the same CRS.")
        if jrc_src.transform != laie_transform:
            raise ValueError("JRC and LAIe rasters must use the same grid transform.")
        jrc_crop = jrc_src.read(1, window=laie_crop_window, masked=True)

    # rgb_path = f"analysis/04a_forest_heterogeneity/temp/rgb_{example_id}.pkl"
    rgb_path = Path(f"temp/rgb_{example_id}.pkl")
    if os.path.exists(rgb_path):
        from pickle import load

        with open(rgb_path, "rb") as f:
            rgb = load(f)
    else:
        rgb = get_sentinel2_rgb(
            geometry_bounds=crop_bounds,
            geometry_crs=laie_crs.to_string(),
            start_date="2020-01-01",
            end_date="2021-01-01",
            dimensions=1500,
        )

        # save rgb pickle
        from pickle import dump

        with open(rgb_path, "wb") as f:
            dump(rgb, f)

    left, bottom, right, top = crop_bounds
    rgb_extent = (left, right, bottom, top)

    # rgb_extent = plotting_extent(rgb, geometry_bbox.getInfo()["coordinates"][0])
    # rgb_image = stretch_rgb(rgb)

    fig = plt.figure(figsize=(14.5, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 5, width_ratios=(1.42, 1.42, 0.07, 1.35, 1.45))
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 3]),
    ]
    panel_d_grid = grid[0, 4].subgridspec(2, 1, height_ratios=(1, 2), hspace=0.12)
    ax_d_mean = fig.add_subplot(panel_d_grid[0, 0])
    ax_d_heterogeneity = fig.add_subplot(panel_d_grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[0, 2])

    # A: RGB context.
    axes[0].imshow(
        rgb,
        extent=rgb_extent,
        origin="upper",
    )
    add_jrc_forest_boundaries(axes[0], jrc_crop, laie_extent)

    add_window_boxes(
        axes[0],
        pair,
        laie_transform,
        args.window_pixels,
    )

    add_scale_bar(axes[0])
    axes[0].set_title("Sentinel-2 RGB (2020)", loc="left", pad=7)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # B: LAIe map on a fixed, publication-wide scale.
    image = axes[1].imshow(
        laie_crop,
        extent=laie_extent,
        origin="upper",
        cmap=LAIE_CMAP,
        norm=LAIE_NORM,
    )
    add_jrc_forest_boundaries(axes[1], jrc_crop, laie_extent)
    add_window_boxes(axes[1], pair, laie_transform, args.window_pixels)
    axes[1].set_title("20 m LAIe", loc="left", pad=7)
    # A dedicated layout column keeps the scale clear of the raster panel.
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="vertical")
    colorbar.ax.tick_params(labelsize=8, length=3)
    colorbar.set_label("LAIe", labelpad=5)

    # C: common-bin distributions within the representative matched pair.
    combined = np.concatenate((plantation_values, control_values))
    bins = np.histogram_bin_edges(combined, bins="fd")
    for values, color, label in (
        (plantation_values, PLANTATION_COLOR, "Plantation"),
        (control_values, CONTROL_COLOR, "Forest"),
    ):
        axes[2].hist(values, bins=bins, density=True, color=color, alpha=0.14)
        axes[2].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=1.8,
            label=(f"{label}\nmean={values.mean():.2f}, SD={values.std(ddof=1):.2f}"),
        )
        axes[2].axvline(
            values.mean(), color=color, linewidth=1.5, linestyle="--", alpha=0.9
        )
    axes[2].set_xlabel("LAIe")
    axes[2].set_ylabel("Density")
    axes[2].set_yticks([])
    axes[2].set_title(f"Representative pair (ID {example_id})", loc="left", pad=7)
    axes[2].legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=0.95,
        fontsize=7.5,
        handlelength=2.2,
    )

    # D: raw paired differences on separate scales for level and heterogeneity.
    rng = np.random.default_rng(args.seed)
    effect_rows = []
    for metric, label in METRICS.items():
        values = metrics[f"difference_{metric}"].dropna().to_numpy(float)
        mean, lower, upper = bootstrap_mean(values, args.bootstrap_samples, rng)
        effect_rows.append((label, mean, lower, upper))

    for ax, rows, y_positions, y_labels, xlabel in (
        (
            ax_d_mean,
            effect_rows[:1],
            np.array([0.0]),
            ["Mean LAIe"],
            "Mean difference (LAIe)",
        ),
        (
            ax_d_heterogeneity,
            effect_rows[1:],
            np.array([0.25, 0.85]),
            ["LAIe SD", "Adjacent\ncontrast"],
            "Heterogeneity difference (LAIe)\nPlantation - Forest",
        ),
    ):
        effects = np.array([row[1] for row in rows])
        lower = np.array([row[2] for row in rows])
        upper = np.array([row[3] for row in rows])
        limit = 1.15 * np.max(np.abs(np.concatenate((lower, upper))))
        ax.errorbar(
            effects,
            y_positions,
            xerr=np.vstack((effects - lower, upper - effects)),
            fmt="o",
            color="#333333",
            ecolor="#666666",
            capsize=3,
            markersize=6,
        )
        ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
        ax.set_yticks(y_positions, y_labels)
        ax.set_xlim(-limit, limit)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for yi, row in zip(y_positions, rows, strict=True):
            ax.annotate(
                f"Δ={row[1]:+.3f}",
                (row[1], yi),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
            )

    ax_d_mean.set_ylim(0.48, -0.38)
    ax_d_heterogeneity.set_ylim(1.25, -0.15)
    ax_d_mean.set_title(f"All matched pairs (n = {len(metrics)})", loc="left", pad=7)

    for label, ax in zip("ABC", axes, strict=True):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        ax.spines[["top", "right"]].set_visible(False)
    ax_d_mean.text(
        0.02,
        0.98,
        "D",
        transform=ax_d_mean.transAxes,
        va="top",
        ha="left",
        fontsize=14,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Representative pair: {example_id}")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
