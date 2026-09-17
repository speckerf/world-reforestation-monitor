#!/usr/bin/env python3
"""Render Figure 4A–C in one Matplotlib figure, using original scientific inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from matplotlib.colors import TwoSlopeNorm
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

# FONT_SIZE_AXES = 7
# FONT_SIZE_LEGEND = 6


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHANGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "change", ["purple", "white", "green"], N=257
)


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
                linewidth=1.0,
            )
        )
        ax.text(
            left,
            top,
            short_label,
            color="white",
            # fontsize=7,
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
        # fontsize=7,
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
            linewidths=0.45,
            linestyles="solid",
            zorder=3,
        )
        boundary.set_path_effects(
            [path_effects.withStroke(linewidth=1.0, foreground="#222222")]
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


def draw_a(fig, args, bottom, height):
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

    rgb_path = ROOT / f"analysis/04a_forest_heterogeneity/temp/rgb_{example_id}.pkl"
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

        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the downloaded RGB for subsequent runs.
        from pickle import dump

        with open(rgb_path, "wb") as f:
            dump(rgb, f)

    left, map_bottom, right, top = crop_bounds
    rgb_extent = (left, right, map_bottom, top)

    axes = [
        panel(fig, x, bottom, width, height)
        for x, width in [(0.04, 0.19), (0.265, 0.19), (0.525, 0.22)]
    ]
    ax_d_mean = panel(fig, 0.76, bottom + height * 0.68, 0.22, height * 0.32)
    ax_d_heterogeneity = panel(fig, 0.76, bottom, 0.22, height * 0.40)
    colorbar_ax = panel(fig, 0.465, bottom, 0.007, height)

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
    # axes[0].set_title("RGB (2020)", loc="left", pad=7)
    axes[0].set_xlabel("RGB (2020)", loc="left")
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
    # axes[1].set_title("20 m LAIe", loc="left", pad=7)
    axes[1].set_xlabel("20 m LAIe")
    # A dedicated layout column keeps the scale clear of the raster panel.
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="vertical")
    colorbar.ax.tick_params(labelsize=7, length=3)
    # colorbar.ax.set_xlabel(r"LAIe", fontsize=FONT_SIZE_AXES, labelpad=5)
    colorbar.ax.set_xlabel(r"LAIe", labelpad=5)
    # set the colorbar label at bottom
    # colorbar.set_label("LAIe", fontsize=6.5, labelpad=5)

    # C: common-bin distributions within the representative matched pair.
    combined = np.concatenate((plantation_values, control_values))
    bins = np.histogram_bin_edges(combined, bins=25)
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
            label=label,
            # label=(f"{label}\nmean={values.mean():.2f}, SD={values.std(ddof=1):.2f}"),
        )
        axes[2].axvline(
            values.mean(), color=color, linewidth=1.0, linestyle="--", alpha=0.9
        )
    axes[2].set_ylim(0, axes[2].get_ylim()[1])
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].set_xlabel("LAIe")
    axes[2].set_ylabel("Density", labelpad=1)
    axes[2].set_yticks([])
    # axes[2].set_title(f"Representative pair (ID {example_id})", loc="left", pad=7)
    axes[2].legend(
        loc="center left",
        bbox_to_anchor=(0.02, 0.8),
        frameon=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=0.95,
        fontsize=6.8,
        handlelength=1.2,
        borderaxespad=0,
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
            "Difference in mean",
        ),
        (
            ax_d_heterogeneity,
            effect_rows[1:],
            np.array([0.25, 0.85]),
            ["LAIe SD", "Adjacent\ncontrast"],
            "Difference (LAIe)",
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
            markersize=3.5,
        )
        ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
        ax.set_yticks([])
        for yi, label in zip(y_positions, y_labels):
            ax.text(
                0.03,
                yi if len(rows) == 1 else yi - 0.14,
                label.replace("\n", " "),
                transform=ax.get_yaxis_transform(),
                va="bottom",
                fontsize=6,
                color="#444444",
            )
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-0.5, 0, 0.5] if len(rows) == 1 else [-0.05, 0, 0.05])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for yi, row in zip(y_positions, rows, strict=True):
            if len(rows) == 1:
                ax.text(
                    0.97,
                    0.10,
                    f"Δ = {row[1]:+.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=6,
                )
            else:
                ax.text(
                    0.98,
                    yi,
                    f"Δ = {row[1]:+.3f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=6,
                )

    ax_d_mean.set_ylim(0.48, -0.38)
    ax_d_heterogeneity.set_ylim(1.25, -0.15)
    ax_d_heterogeneity.set_xlabel(
        "Difference in heterogeneity\nPlantation − Forest (Δ)", fontsize=7
    )
    ax_d_mean.set_title(f"All matched pairs (n = {len(metrics)})", loc="left", pad=7)

    for label, ax in zip(("A1", "A2", "A3", "A4"), [*axes, ax_d_mean]):
        panel_label(ax, label)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2].set_title("Representative pair", loc="left")
    # ax_d_mean.set_title(f"{len(metrics)} matched pairs", loc="left")
    if len(metrics) != 464:
        raise ValueError(f"Expected 464 valid matched pairs; found {len(metrics)}")


def panel(fig, x, y, width, height):
    return fig.add_axes([x, y, width, height])


def panel_label(ax, text):
    ax.text(
        0.025,
        0.975,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        weight="bold",
        color="black",
        zorder=20,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )


def load_campeche(args):
    """Cache unannotated map rasters and original 100 m annual statistics."""
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = cache_dir / "campeche.json"
    paths = [cache_dir / f"campeche_{name}.png" for name in ("2020", "2025", "change")]
    if metadata.exists() and all(p.exists() for p in paths) and not args.refresh:
        return json.loads(metadata.read_text()), [
            np.asarray(Image.open(p)) for p in paths
        ]
    # Use the notebook's geometry, asset versions, scaling, and annual reducer.
    sys.path.insert(0, str(ROOT / "revision-figures"))
    from utils.plotting_utils import get_geometries, get_imgc, get_time_series

    ee.Initialize(project=args.ee_project)
    geom = get_geometries()["campeche"]
    bounds = geom.buffer(300).bounds()
    ring = bounds.coordinates().getInfo()[0]
    extent = [
        min(p[0] for p in ring),
        max(p[0] for p in ring),
        min(p[1] for p in ring),
        max(p[1] for p in ring),
    ]
    pre, post = [
        get_imgc("laie", year=y, resolution="20m", version="v03")
        .filterBounds(bounds)
        .first()
        .select("laie_mean")
        .divide(1000)
        for y in (2020, 2025)
    ]
    images = []
    for img, path, vis in zip(
        [pre, post, post.subtract(pre)],
        paths,
        [dict(min=0, max=5, palette=LAIE_COLORS)] * 2
        + [dict(min=-2, max=2, palette=["800080", "ffffff", "008000"])],
    ):
        url = img.visualize(**vis).getThumbURL(
            dict(region=bounds, dimensions=1500, format="png")
        )
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        image.save(path)
        images.append(np.asarray(image))
    series = {}
    for trait in ("laie", "fcover", "fapar"):
        ts = get_time_series(trait, geom, resolution="20m")
        ts = ts.loc[ts.index <= 2025]
        series[trait] = dict(
            year=ts.index.tolist(),
            mean=ts[f"{trait}_mean"].tolist(),
            sd=ts[f"{trait}_stdDev"].tolist(),
        )
    data = dict(extent=extent, geometry=geom.getInfo(), series=series)
    metadata.write_text(json.dumps(data, indent=2))
    return data, images


def draw_b(fig, args, bottom, height):
    from shapely.geometry import shape

    data, images = load_campeche(args)
    axes = [panel(fig, x, bottom, 0.185, height) for x in (0.04, 0.27, 0.50)]
    for i, (ax, pixels, title) in enumerate(
        zip(axes, images, ["LAIe (2020)", "LAIe (2025)", "Change (2025 − 2020)"])
    ):
        ax.imshow(pixels, extent=data["extent"], interpolation="nearest")
        gpd.GeoSeries([shape(data["geometry"])]).boundary.plot(
            ax=ax, color="#704214", linewidth=0.65
        )
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(data["extent"][2:]))))
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(title, loc="left")
        ax.set_xlabel(title)
        panel_label(ax, f"B{i + 1}")
    # Ellipsoidal geodesic distance, rather than a hard-coded pixel length.
    from pyproj import Geod

    left, right, lower, upper = data["extent"]
    x, y = left + 0.06 * (right - left), lower + 0.08 * (upper - lower)
    x2, _, _ = Geod(ellps="WGS84").fwd(x, y, 90, 500)
    axes[0].plot([x, x2], [y, y], color="white", lw=2)
    axes[0].text(
        (x + x2) / 2,
        y + 0.025 * (upper - lower),
        "500 m",
        color="white",
        ha="center",
        fontsize=7,
        path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
    )
    for x, cmap, norm, label in [
        (0.465, LAIE_CMAP, LAIE_NORM, "LAIe"),
        (0.695, CHANGE_CMAP, TwoSlopeNorm(0, -2, 2), "Δ LAIe"),
    ]:
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=panel(fig, x, bottom, 0.007, height),
            ticks=[0, 1, 2, 3, 4, 5] if label == "LAIe" else [-2, -1, 0, 1, 2],
        )
        # cb.ax.set_title(label, fontsize=6.5, pad=5)
        cb.ax.set_xlabel(
            label,
        )
    ax = panel(fig, 0.78, bottom, 0.15, height)
    right_ax = ax.twinx()
    for trait, color, target in [
        ("laie", "#172313", ax),
        ("fcover", "#74c476", right_ax),
        ("fapar", "#c53859", right_ax),
    ]:
        ts = data["series"][trait]
        year, mean, sd = [np.asarray(ts[k]) for k in ("year", "mean", "sd")]
        target.plot(
            year,
            mean,
            color=color,
            marker="o",
            markersize=2.5,
            label=trait.upper() if trait != "laie" else "LAIe",
        )
        target.fill_between(
            year, mean - sd, mean + sd, color=color, alpha=0.1, linewidth=0
        )
    ax.axvline(2021, color="grey", linestyle="--", lw=0.7)
    ax.set(
        ylim=(0, 4),
        ylabel="LAIe",
        # xlabel="Year",
        xticks=[2019, 2022, 2025],
        yticks=[0, 1, 2, 3, 4],
    )
    right_ax.set(ylim=(0, 1), ylabel="FAPAR / FCOVER", yticks=[0, 0.5, 1])
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = right_ax.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="lower right", fontsize=6)
    # ax.set_title("Annual trajectory", loc="left")
    panel_label(ax, "B4")


def draw_c(fig, args, bottom, height):
    import importlib.util

    source = ROOT / "analysis/04b_forest_disturbance/analysis"
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "harvest_row", source / "03_create_figure4_row_c_final.py"
    )
    harvest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harvest)
    harvest.DIFFERENCE_CMAP = CHANGE_CMAP
    polygons = (
        gpd.read_file(harvest.POLYGON_PATH)
        .to_crs(harvest.EPSG_EXAMPLES)
        .set_index("disturbance_id")
    )
    # C1–C3 are resolution columns, each containing both fixed examples.
    for j, (resolution, x) in enumerate(zip([20, 100, 300], [0.08, 0.25, 0.42])):
        for i, (identifier, name) in enumerate(
            zip(harvest.DISTURBANCE_IDS, ["Small harvest", "Large harvest"])
        ):
            polygon = polygons.loc[identifier]
            y = bottom + (height * 0.51 if i == 0 else 0)
            ax = panel(fig, x, y, 0.145, height * 0.49)
            image = harvest.plot_difference_map(
                ax,
                polygon,
                resolution,
                harvest.adaptive_map_bounds(polygon.geometry),
                harvest.EPSG_EXAMPLES,
                harvest.MGRS_EXAMPLES[i],
                add_ruler=(j == 0),
            )
            image.set_norm(TwoSlopeNorm(0, -0.6, 0.6))

            ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.6,
                    clip_on=False,
                    zorder=20,
                )
            )
            for text in ax.texts:
                text.set_fontsize(6.5)
                text.set_fontweight("normal")
            if i == 0:
                fig.text(
                    x,
                    bottom + height + 2 / 214,
                    f"{resolution} m",
                    fontsize=7,
                    va="bottom",
                )
                panel_label(ax, f"C{j + 1}")
            if j == 0:
                ax.text(
                    -0.13,
                    0.5,
                    f"{name}\n({polygon['area_ha']:.1f} ha)",
                    rotation=90,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    # fontsize=6.0,
                )
    cb = fig.colorbar(
        image,
        cax=panel(fig, 0.570, bottom, 0.007, height),
        ticks=[-0.6, -0.3, 0, 0.3, 0.6],
    )
    # cb.ax.set_title("Δ FCOVER", fontsize=6.5, pad=5)
    ax = panel(fig, 0.695, bottom, 0.265, height)
    harvest.plot_summary(ax, pd.read_csv(harvest.SUMMARY_PATH))
    ax.set_title("National results by harvest size", loc="left", fontsize=7)
    ax.set_ylabel("Δ FCOVER (post − pre)", fontsize=7)
    ax.set_xlabel("Resolution (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    for line in ax.lines:
        line.set_markersize(3)
        line.set_linewidth(0.9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Harvest area",
        fontsize=6,
        title_fontsize=7,
        frameon=False,
        loc="lower right",
    )
    panel_label(ax, "C4")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "analysis/04a_forest_heterogeneity"
    for name, default in {
        "laie": base / "data/laie_tasmania/tasmania_laie_2020_20m.tif",
        "jrc": base / "data/JRC_GFT2020_V1_tasmania_match_laie20.tif",
        "pairs": base / "results/forest_window_matched_pairs.gpkg",
        "metrics": base / "results/matched_pair_laie_heterogeneity.csv",
        "output": HERE / "figure_4",
        "cache-dir": HERE / "cache",
    }.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--ee-project", default="ee-speckerfelix")
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh Campeche Earth Engine cache"
    )
    parser.add_argument("--pair-id", type=int, default=380)
    parser.add_argument("--pairs-layer", default="matched_pairs")
    parser.add_argument("--window-pixels", type=int, default=25)
    parser.add_argument("--padding-m", type=float, default=300)
    parser.add_argument("--laie-scale-factor", type=float, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dpi", type=int, default=500, help="Raster resolution; canvas is 190 × 214 mm"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_samples < 1 or args.window_pixels < 1 or args.dpi < 1:
        raise ValueError("Bootstrap samples, window pixels, and DPI must be positive")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(190 / 25.4, 214 / 25.4))
    # Fixed canvas: a 2.5 mm gap separates equal-width row containers.
    rows_mm = [
        (143.0, 66, 12, 42),  # A
        (87.5, 53, 9, 33),  # B: bottom +4, height −4, inset −4
        (7.0, 78, 11, 54),  # C: shift upward by 4
    ]
    titles = [
        "Landscape heterogeneity / Tasmania, Australia",
        "Restoration monitoring / Campeche, Mexico",
        "Forest-harvest signals across resolutions / Sweden",
    ]
    for i, (letter, title, draw) in enumerate(
        zip("ABC", titles, [draw_a, draw_b, draw_c])
    ):
        bottom_mm, height_mm, inset_mm, panel_height_mm = rows_mm[i]
        bottom = bottom_mm / 214
        row_height = height_mm / 214
        top = bottom + row_height
        fig.add_artist(
            Rectangle(
                (0.015, bottom),
                0.97,
                row_height,
                transform=fig.transFigure,
                facecolor="#F7F7F7",
                edgecolor="#BDBDBD",
                linewidth=0.6,
                zorder=-10,
            )
        )
        fig.text(
            0.03,
            top - 0.012,
            f"{letter}   {title}",
            va="top",
            fontsize=8.5,
            weight="bold",
        )
        print(f"Rendering Figure 4{letter}", flush=True)
        draw(fig, args, bottom + inset_mm / 214, panel_height_mm / 214)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for index, ax in enumerate(fig.axes):
        bounds = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > 1 or bounds.y1 > 1:
            raise RuntimeError(
                f"Axis {index} extends beyond the export canvas: {bounds.bounds}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        path = args.output.with_suffix(suffix)
        fig.savefig(path, dpi=args.dpi)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
