from pathlib import Path

import ee
import rioxarray  # noqa: F401
import xarray as xr
import xee  # noqa: F401
from affine import Affine
from shapely.geometry import box
from xee.helpers import fit_geometry

ee.Initialize(
    project="gem-eth-analysis",
    opt_url="https://earthengine-highvolume.googleapis.com",
)

ASSET_ID = "projects/ee-speckerfelix/assets/open-earth/laie_predictions-mlp_20m_v03"

CRS = "EPSG:32755"
YEAR = 2020
RESOLUTIONS = (20, 100, 300)

# Geographic bounding box: west, south, east, north
BBOX = box(143.5, -44.0, 148.5, -39.5)
BBOX_EE = ee.Geometry.BBox(*BBOX.bounds)


def make_grid(resolution: int) -> dict:
    """Create an EPSG:32755 grid fitted to the geographic bounding box."""

    return fit_geometry(
        geometry=BBOX,
        geometry_crs="EPSG:4326",
        grid_crs=CRS,
        grid_scale=(resolution, -resolution),
    )


def grid_projection(grid: dict) -> ee.Projection:
    """Convert xee grid parameters to an Earth Engine projection."""

    return ee.Projection(
        crs=grid["crs"],
        transform=list(grid["crs_transform"]),
    )


def get_laie_20m(grid: dict) -> ee.Image:
    """Load and scale the 2020 20 m LAIe mean image."""

    return (
        ee.ImageCollection(ASSET_ID)
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .filterBounds(BBOX_EE)
        .mosaic()
        .select("laie_mean")
        .divide(1000)
        .rename("laie_mean")
        .setDefaultProjection(grid_projection(grid))
        .clip(BBOX_EE)
    )


def aggregate_mean(
    image: ee.Image,
    grid: dict,
) -> ee.Image:
    """Aggregate the 20 m image to a coarser fitted grid."""

    return (
        image.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=1024,
        )
        .reproject(grid_projection(grid))
        .rename("laie_mean")
    )


def image_to_xarray(
    image: ee.Image,
    grid: dict,
    resolution: int,
) -> xr.Dataset:
    """Open one Earth Engine image as a lazy xarray Dataset."""

    image = image.set(
        {
            "system:index": f"laie_{YEAR}_{resolution}m",
            "system:time_start": ee.Date(f"{YEAR}-01-01").millis(),
        }
    )

    collection = ee.ImageCollection.fromImages([image])

    dataset = xr.open_dataset(
        collection,
        engine="ee",
        **grid,
    )

    dataset.attrs.update(
        {
            "crs": CRS,
            "resolution_m": resolution,
            "year": YEAR,
        }
    )

    return dataset


def load_laie_datasets() -> tuple[
    dict[int, xr.Dataset],
    dict[int, dict],
]:
    grids = {resolution: make_grid(resolution) for resolution in RESOLUTIONS}

    laie20 = get_laie_20m(grids[20])

    images = {
        20: laie20,
        100: aggregate_mean(laie20, grids[100]),
        300: aggregate_mean(laie20, grids[300]),
    }

    datasets = {
        resolution: image_to_xarray(
            image=images[resolution],
            grid=grids[resolution],
            resolution=resolution,
        )
        for resolution in RESOLUTIONS
    }

    return datasets, grids


def save_dataset_as_tif(
    dataset: xr.Dataset,
    grid: dict,
    output_path: Path,
) -> None:
    """Fetch a complete xee dataset and save it as a GeoTIFF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = dataset["laie_mean"]

    if "time" in data.dims:
        data = data.isel(time=0, drop=True)

    data = (
        data.rio.set_spatial_dims(
            x_dim="x",
            y_dim="y",
        )
        .rio.write_crs(grid["crs"])
        .rio.write_transform(Affine(*grid["crs_transform"]))
        .astype("float32")
        .compute()
    )

    data.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        BIGTIFF="YES",
        dtype="float32",
    )


def main() -> None:
    datasets, grids = load_laie_datasets()

    ds20 = datasets[20]
    ds100 = datasets[100]
    ds300 = datasets[300]

    print(ds20)
    print(ds100)
    print(ds300)

    for resolution in reversed(RESOLUTIONS):
        output_path = (
            Path("analysis", "04a_forest_heterogeneity", "data/laie_tasmania")
            / f"tasmania_laie_{YEAR}_{resolution}m.tif"
        )

        save_dataset_as_tif(
            dataset=datasets[resolution],
            grid=grids[resolution],
            output_path=output_path,
        )

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
