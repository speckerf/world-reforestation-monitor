"""Plot pre/post-intervention FCOVER maps for ten disturbance polygons."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray
from matplotlib.backends.backend_pdf import PdfPages
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from shapely.geometry import box

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_DIR / "data"
FCOVER_DIR = DATA_DIR / "fcover_sweden"
POLYGON_PATH = DATA_DIR / "sweden_completed_harvests_2020_2024.gpkg"
OUTPUT_DIR = ANALYSIS_DIR / "figures" / "fcover_examples"


def fcover_path(year: int, resolution_m: int, tile: str, epsg_code: str) -> Path:
    """Return the expected path for an annual FCOVER raster."""

    if "epsg" in epsg_code.lower():
        epsg_code = epsg_code.split(":")[-1]

    path = (
        FCOVER_DIR / f"sweden_fcover_{year}_{resolution_m}m_epsg-{epsg_code}_{tile}.tif"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing FCOVER raster: {path}")
    return path


def select_examples(
    polygons: gpd.GeoDataFrame,
    raster_crs,
    raster_bounds,
    n_examples: int,
    random_seed: int,
) -> gpd.GeoDataFrame:
    """Select examples across disturbance years within the raster footprint."""

    polygons = polygons[
        polygons.geometry.notna()
        & ~polygons.geometry.is_empty
        & ~polygons.geometry.type.isin(["GeometryCollection"])
    ].to_crs(raster_crs)
    polygons = polygons[polygons.geometry.intersects(box(*raster_bounds))].copy()
    polygons["disturbance_year"] = polygons["disturbance_year"].astype(int)

    years = sorted(polygons["disturbance_year"].unique())
    if not years:
        raise RuntimeError("No disturbance polygons intersect the FCOVER rasters")

    selected_indices = []
    base, remainder = divmod(n_examples, len(years))
    for position, year in enumerate(years):
        n_for_year = base + (position < remainder)
        candidates = polygons[polygons["disturbance_year"] == year]
        n_for_year = min(n_for_year, len(candidates))
        selected_indices.extend(
            candidates.sample(n=n_for_year, random_state=random_seed + year).index
        )

    if len(selected_indices) < n_examples:
        remaining = polygons.drop(index=selected_indices)
        n_remaining = min(n_examples - len(selected_indices), len(remaining))
        selected_indices.extend(
            remaining.sample(n=n_remaining, random_state=random_seed).index
        )

    if len(selected_indices) < n_examples:
        raise RuntimeError(
            f"Requested {n_examples} examples, but only {len(selected_indices)} "
            "eligible polygons are available"
        )

    return polygons.loc[selected_indices].sort_values(
        ["disturbance_year", "disturbance_id"]
    )


def map_bounds(geometry, context_m: float) -> tuple[float, float, float, float]:
    """Create a square map extent around a polygon."""

    minx, miny, maxx, maxy = geometry.bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    half_size = max(context_m, 0.75 * max(maxx - minx, maxy - miny))
    return (
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size,
    )


def read_pair_with_grid(
    pre_path: str | Path,
    post_path: str | Path,
    bounds: tuple[float, float, float, float],
):
    """
    Read matching pre/post raster windows and return grid metadata.

    Parameters
    ----------
    pre_path, post_path
        Paths to rasters on the same grid.
    bounds
        (xmin, ymin, xmax, ymax) in the raster CRS.

    Returns
    -------
    pre : np.ndarray
    post : np.ndarray
    grid : dict
        Contains shape, transform, crs, and plotting extent.
    """

    xmin, ymin, xmax, ymax = bounds

    with rasterio.open(pre_path) as pre_src, rasterio.open(post_path) as post_src:
        if pre_src.crs != post_src.crs:
            raise ValueError(f"CRS mismatch: {pre_src.crs} != {post_src.crs}")

        if pre_src.transform != post_src.transform:
            raise ValueError("Pre/post rasters are not on the same grid.")

        window = from_bounds(
            xmin,
            ymin,
            xmax,
            ymax,
            transform=pre_src.transform,
        )

        # Snap to integer source-pixel boundaries.
        window = window.round_offsets().round_lengths()

        pre = pre_src.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan,
        ).astype(np.float32)

        post = post_src.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan,
        ).astype(np.float32)

        transform = pre_src.window_transform(window)

        # Convert nodata to NaN if required.
        if pre_src.nodata is not None:
            pre[pre == pre_src.nodata] = np.nan

        if post_src.nodata is not None:
            post[post == post_src.nodata] = np.nan

        height, width = pre.shape

        left = transform.c
        top = transform.f
        right = left + width * transform.a
        bottom = top + height * transform.e

        extent = [
            min(left, right),
            max(left, right),
            min(bottom, top),
            max(bottom, top),
        ]

        grid = {
            "shape": pre.shape,
            "transform": transform,
            "crs": pre_src.crs,
            "extent": extent,
        }

    return pre, post, grid


def polygon_mean(values, geometry, transform) -> float:
    """Calculate the raster mean within a displayed disturbance polygon."""

    inside = geometry_mask(
        [geometry],
        out_shape=values.shape,
        transform=transform,
        invert=True,
    )
    data = np.ma.array(values, mask=np.ma.getmaskarray(values) | ~inside)
    return float(data.mean()) if data.count() else np.nan


def read_on_template(
    source_path,
    *,
    template_shape,
    template_transform,
) -> np.ndarray:
    """Read only the required source window and map it onto the template grid."""

    template_bounds = rasterio.transform.array_bounds(
        template_shape[0],
        template_shape[1],
        template_transform,
    )

    with rioxarray.open_rasterio(
        source_path,
        masked=True,
        mask_and_scale=True,
    ) as da:
        res_x, res_y = da.rio.resolution()
        pad = max(abs(res_x), abs(res_y))

        # Small native-grid subset only
        da = da.rio.clip_box(
            minx=template_bounds[0] - pad,
            miny=template_bounds[1] - pad,
            maxx=template_bounds[2] + pad,
            maxy=template_bounds[3] + pad,
        )

        # Reproject only this tiny subset
        da = da.rio.reproject(
            dst_crs=da.rio.crs,
            shape=template_shape,
            transform=template_transform,
            resampling=Resampling.nearest,
            nodata=np.nan,
        )

        return da.squeeze("band", drop=True).values.astype(np.float32)


def plot_example(
    polygon,
    example_number: int,
    resolution_m: int,
    context_m: float,
):
    """Build one pre, post, and difference map figure."""

    year = int(polygon["disturbance_year"])
    pre_year, post_year = year - 1, year + 1
    pre_path = fcover_path(pre_year, resolution_m, tile, epsg_code)
    post_path = fcover_path(post_year, resolution_m, tile, epsg_code)
    bounds = map_bounds(polygon.geometry, context_m)
    pre, post, grid = read_pair_with_grid(pre_path, post_path, bounds)
    shape, transform, crs, extent = (
        grid["shape"],
        grid["transform"],
        grid["crs"],
        grid["extent"],
    )
    difference = post - pre

    pre_mean = polygon_mean(pre, polygon.geometry, transform)
    post_mean = polygon_mean(post, polygon.geometry, transform)
    difference_mean = post_mean - pre_mean

    valid_difference = difference.compressed()
    difference_limit = (
        float(np.nanpercentile(np.abs(valid_difference), 98))
        if valid_difference.size
        else 0.1
    )
    difference_limit = max(difference_limit, 0.05)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    pre_image = axes[0].imshow(
        pre,
        extent=extent,
        origin="upper",
        cmap="YlGn",
        vmin=0,
        vmax=1,
    )
    axes[1].imshow(
        post,
        extent=extent,
        origin="upper",
        cmap="YlGn",
        vmin=0,
        vmax=1,
    )
    difference_image = axes[2].imshow(
        difference,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )

    titles = (
        f"Pre-intervention ({pre_year})\npolygon mean = {pre_mean:.3f}",
        f"Post-intervention ({post_year})\npolygon mean = {post_mean:.3f}",
        f"Difference: post − pre\npolygon mean = {difference_mean:+.3f}",
    )
    boundary = gpd.GeoSeries([polygon.geometry]).boundary
    for axis, title in zip(axes, titles):
        boundary.plot(ax=axis, color="black", linewidth=1.5)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_axis_off()

    figure.colorbar(pre_image, ax=axes[:2], shrink=0.75, label="FCOVER")
    figure.colorbar(
        difference_image,
        ax=axes[2],
        shrink=0.75,
        label="FCOVER difference",
    )
    figure.suptitle(
        f"Example {example_number}: disturbance {polygon['disturbance_id']} "
        f"({year}, {resolution_m} m)",
        fontsize=14,
    )
    return figure


def main(
    n_examples: int = 10,
    resolution_m: int = 20,
    context_m: float = 500,
    random_seed: int = 42,
) -> None:
    """Create individual PNG maps and a combined multipage PDF."""

    polygons = gpd.read_file(POLYGON_PATH)
    reference_path = fcover_path(
        int(polygons["disturbance_year"].min()) - 1,
        resolution_m,
    )
    with rasterio.open(reference_path) as reference:
        examples = select_examples(
            polygons,
            reference.crs,
            reference.bounds,
            n_examples,
            random_seed,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"fcover_{resolution_m}m_pre_post_examples.pdf"
    with PdfPages(pdf_path) as pdf:
        for example_number, (_, polygon) in enumerate(examples.iterrows(), start=1):
            figure = plot_example(
                polygon,
                example_number,
                resolution_m,
                context_m,
            )
            disturbance_id = str(polygon["disturbance_id"]).replace("/", "-")
            png_path = OUTPUT_DIR / (
                f"example_{example_number:02d}_{disturbance_id}_"
                f"{int(polygon['disturbance_year'])}.png"
            )
            figure.savefig(png_path, dpi=200, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {png_path}")

    print(f"Saved combined PDF: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--resolution-m", type=int, default=20)
    parser.add_argument("--context-m", type=float, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    arguments = parser.parse_args()
    main(**vars(arguments))
