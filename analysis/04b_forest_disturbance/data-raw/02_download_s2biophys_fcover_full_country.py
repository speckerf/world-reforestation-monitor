import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import geopandas as gpd
import rioxarray  # noqa: F401
import shapely
import xarray as xr
import xee  # noqa: F401
from affine import Affine
from loguru import logger
from shapely.geometry import box
from xee.helpers import extract_grid_params

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ee.Initialize(
    project="gem-eth-analysis",
    opt_url="https://earthengine-highvolume.googleapis.com",
)

ASSET_ID = "projects/ee-speckerfelix/assets/open-earth/fcover_predictions-mlp_20m_v03"

BAND_NAME = "fcover_mean"
VARIABLE_NAME = "fcover"

RESOLUTION = 20

OUTPUT_DIR = Path(
    "analysis",
    "04b_forest_disturbance",
    "data",
    "fcover_sweden",
)

MAX_WORKERS = 4


# ---------------------------------------------------------------------
# xarray conversion and export
# ---------------------------------------------------------------------


def image_to_xarray(
    image: ee.Image,
    grid: dict,
    *,
    year: int,
    resolution: int,
) -> xr.Dataset:
    """Open one Earth Engine image lazily as an xarray Dataset."""

    dataset = xr.open_dataset(
        image,
        engine="ee",
        **grid,
    )

    dataset.attrs.update(
        {
            "crs": grid["crs"],
            "resolution_m": resolution,
            "year": year,
            "source_asset": ASSET_ID,
            "variable": VARIABLE_NAME,
        }
    )

    return dataset


def save_dataset_as_tif(
    dataset: xr.Dataset,
    grid: dict,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Fetch an xee dataset and write FCOVER as scaled int16."""

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = dataset[VARIABLE_NAME]

    if "time" in data.dims:
        data = data.isel(time=0, drop=True)

    nodata = -9999

    data = (
        data.rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_crs(grid["crs"])
        .rio.write_transform(Affine(*grid["crs_transform"]))
    )

    # FCOVER [0, 1] -> integer [0, 10000]
    data = (data * 10000).round().fillna(nodata).astype("int16").compute()

    data.rio.write_nodata(nodata, inplace=True)

    data.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
        dtype="int16",
        nodata=nodata,
    )


def crop_grid_to_geometry(
    grid: dict,
    geometry: shapely.geometry.base.BaseGeometry,
    geometry_crs,
) -> dict:
    """
    Crop an xee grid to the bounding box of its intersection with a geometry.

    The resulting grid remains exactly aligned with the original raster grid.
    """

    crs = grid["crs"]

    # Project country boundary to the image CRS
    geometry_projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(crs).iloc[0]

    a, b, c, d, e, f = grid["crs_transform"]
    width, height = grid["shape_2d"]

    if b != 0 or d != 0:
        raise ValueError("Only north-up grids are currently supported.")

    # Full raster footprint
    xmin = c
    xmax = c + width * a
    ymax = f
    ymin = f + height * e

    grid_geometry = box(
        min(xmin, xmax),
        min(ymin, ymax),
        max(xmin, xmax),
        max(ymin, ymax),
    )

    intersection = grid_geometry.intersection(geometry_projected)

    if intersection.is_empty:
        logger.warning("Image grid does not intersect the supplied geometry.")
        # return empty grid with original CRS and transform, but no pixels
        return {
            "crs": crs,
            "crs_transform": grid["crs_transform"],
            "shape_2d": (0, 0),
        }

    minx, miny, maxx, maxy = intersection.bounds

    # -------------------------------------------------------------
    # Snap intersection bounds to ORIGINAL pixel grid
    # -------------------------------------------------------------

    pixel_width = a
    pixel_height = abs(e)

    col_start = math.floor((minx - c) / pixel_width)
    col_stop = math.ceil((maxx - c) / pixel_width)

    row_start = math.floor((f - maxy) / pixel_height)
    row_stop = math.ceil((f - miny) / pixel_height)

    # Keep window within original image
    col_start = max(0, col_start)
    row_start = max(0, row_start)

    col_stop = min(width, col_stop)
    row_stop = min(height, row_stop)

    cropped_width = col_stop - col_start
    cropped_height = row_stop - row_start

    # New upper-left corner, still aligned to source grid
    new_c = c + col_start * a
    new_f = f + row_start * e

    return {
        "crs": crs,
        "crs_transform": (
            a,
            b,
            new_c,
            d,
            e,
            new_f,
        ),
        "shape_2d": (
            cropped_width,
            cropped_height,
        ),
    }


# ---------------------------------------------------------------------
# Sweden geometry
# ---------------------------------------------------------------------


def load_sweden_geometry() -> tuple[
    gpd.GeoDataFrame,
    ee.Geometry,
]:
    """Load Sweden geometry locally and from Earth Engine."""

    geoboundaries_path_local = Path(
        "analysis",
        "04b_forest_disturbance",
        "data-raw",
        "geoBoundaries-SWE-ADM0-all",
        "geoBoundaries-SWE-ADM0_simplified.shp",
    )

    sweden_local = gpd.read_file(geoboundaries_path_local)

    geoboundaries_fc = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0")

    sweden_ee = (
        geoboundaries_fc.filter(ee.Filter.eq("shapeName", "Sweden")).first().geometry()
    )

    return sweden_local, sweden_ee


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def process_image(
    year: int,
    index: str,
    sweden_ee: ee.Geometry,
    sweden_geometry: shapely.geometry.base.BaseGeometry,
    sweden_crs: str,
) -> Path | None:
    """Download and save one FCOVER image intersecting Sweden."""

    print(f"Processing {year}: {index}")

    img = (
        ee.Image(f"{ASSET_ID}/{index}")
        .select(BAND_NAME)
        .divide(10000)
        .rename(VARIABLE_NAME)
        .clip(sweden_ee)
    )

    # Native image grid
    grid_full = extract_grid_params(img)

    # Restrict request to Sweden intersection
    grid = crop_grid_to_geometry(
        grid=grid_full,
        geometry=sweden_geometry,
        geometry_crs=sweden_crs,
    )

    width, height = grid["shape_2d"]

    if width == 0 or height == 0:
        print(f"Skipping {index}: no intersection with Sweden")
        return None

    parts = index.split("_")
    crs = parts[-2]
    mgrs = parts[-3][1:]

    output_path = OUTPUT_DIR / f"sweden_fcover_{year}_20m_{crs}_{mgrs}.tif"

    # Useful when restarting
    if output_path.exists():
        print(f"Skipping existing: {output_path}")
        return output_path

    dataset = image_to_xarray(
        image=img,
        grid=grid,
        year=year,
        resolution=RESOLUTION,
    )

    save_dataset_as_tif(
        dataset=dataset,
        grid=grid,
        output_path=output_path,
    )

    print(f"Finished {year}: {index}")

    return output_path


def main() -> None:
    sweden_gpd, sweden_ee = load_sweden_geometry()

    sweden_geometry = sweden_gpd.geometry.iloc[0]
    sweden_crs = sweden_gpd.crs

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []

    # First collect all year/image combinations
    for year in [2019, 2025]:  # range(2020, 2025):
        imgc = (
            ee.ImageCollection(ASSET_ID)
            .filterDate(
                f"{year}-01-01",
                f"{year + 1}-01-01",
            )
            .filterBounds(sweden_ee)
        )

        indices = imgc.aggregate_array("system:index").getInfo()

        print(f"{year}: {len(indices)} images")

        jobs.extend((year, index) for index in indices)

    print(f"Total images: {len(jobs)}")

    # Process tiles concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_image,
                year,
                index,
                sweden_ee,
                sweden_geometry,
                sweden_crs,
            ): (year, index)
            for year, index in jobs
        }

        for future in as_completed(futures):
            year, index = futures[future]

            try:
                future.result()

            except Exception as exc:
                print(f"FAILED {year}: {index}\n  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
