import concurrent.futures
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import ee
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from loguru import logger
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xee import helpers

from analysis.domain_representativeness_experiment.prepare_lut_domain import (
    load_lut_for_trait,
)
from ee_translator.ee_standard_scaler import eeStandardScaler
from gee_pipeline.utilsCloudfree import apply_cloudScorePlus_mask

BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
N_NEAREST_NEIGHBORS = 5
MAX_RESCALE_DISTANCE = (
    10  # Maximum distance to consider for rescaling (for uint8 storage)
)


def load_s2_imgc_mgrs(mgrs_tile: str) -> List[str]:
    """Load Sentinel-2 image indices for a given MGRS tile."""
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    folder = Path("data", "gee_pipeline", "outputs", "s2_indices_per_mgrs_tile")
    filenames_s2_indices = [
        folder / f"s2-indices_{year}_mgrs-tile-{mgrs_tile}_v03.txt" for year in years
    ]

    s2_indices = [open(f).read().splitlines() for f in filenames_s2_indices]
    # flatten list of lists
    s2_indices = [item for sublist in s2_indices for item in sublist]
    return s2_indices


def random_sample_s2_indices(
    mgrs_s2_indices: List[str], sample_prop: float = 0.25, seed: int = 42
) -> List[str]:
    """Randomly sample S2 indices by tile with given seed."""
    random.seed(seed)

    # group s2 indices by tile
    s2_indices_by_tile = {}
    for idx in mgrs_s2_indices:
        tile = idx.split("_")[-1].split("-")[-1]
        if tile not in s2_indices_by_tile:
            s2_indices_by_tile[tile] = []
        s2_indices_by_tile[tile].append(idx)

    # sample from same tile with prop sample_prop
    sampled_s2_indices = []
    for tile, indices in s2_indices_by_tile.items():
        n_sample = int(len(indices) * sample_prop)
        if n_sample > 0:
            sampled_indices = random.sample(indices, n_sample)
            sampled_s2_indices.extend(sampled_indices)

    return sampled_s2_indices


def get_mgrs_tiles() -> List[str]:
    raise NotImplementedError(
        "This function is not needed anymore since we load the list of MGRS tiles from precomputed CSV"
    )
    """Load all MGRS tiles from the CSV like in srcGlobal.py."""
    mgrs_tiles_df = pd.read_csv(
        os.path.join(
            "data",
            "gee_pipeline",
            "outputs",
            "mgrs_tiles",
            "mgrs_tiles_all_land_ecoregions.csv",
        )
    )
    mgrs_tiles_list = list(set(mgrs_tiles_df["mgrs_tile_3"].tolist()))
    return mgrs_tiles_list


def get_mgrs_from_ref() -> List[str]:
    """Get list of MGRS tiles from the reference image collection."""
    ref_imgc = ee.ImageCollection(
        "projects/ee-speckerfelix/assets/open-earth/laie_predictions-mlp_20m_v03"
    )
    mgrs_tiles = ref_imgc.aggregate_array("mgrs_tile").distinct().getInfo()
    return mgrs_tiles


def _apply_water_and_rock_mask(img: ee.Image) -> ee.Image:
    # mask out all water pixels: ESA WorldCover class 80 ("Water")
    water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
    img = img.updateMask(water_mask_2020.neq(80))

    # mask out all rock and ice pixels: ECO_ID = 0 ("Rock and Ice")
    ecoregions = ee.Image(
        "projects/ee-speckerfelix/assets/open-earth/resolve_ecoregions_2017_rasterized"
    ).select("Resolve_Ecoregion")
    img = img.updateMask(ecoregions.neq(0))

    return img


def imagecollection_to_transform_fc(imgc, band=None):
    """
    Convert an Earth Engine ImageCollection to a FeatureCollection containing
    the 6 affine transform parameters of each image's projection.

    Output properties per feature:
      - elt_0_0
      - elt_0_1
      - elt_0_2
      - elt_1_0
      - elt_1_1
      - elt_1_2

    Additional metadata:
      - system:index
      - system:time_start
      - band

    Parameters
    ----------
    imgc : ee.ImageCollection
        Input image collection.
    band : str or None
        Band whose projection should be used. If None, uses the first band.

    Returns
    -------
    ee.FeatureCollection
    """

    def _extract_param(transform_str, name):
        transform_str = ee.String(transform_str)
        pattern = ee.String('.*PARAMETER\\["').cat(name).cat('",\\s*([-0-9.]+)\\].*')
        value = transform_str.replace(pattern, "$1")
        return ee.Number.parse(value)

    def _img_to_feature(img):
        img = ee.Image(img)

        proj = img.select(["B2"]).projection()
        tstr = ee.String(proj.transform())

        transform_lst = ee.List(
            [
                ee.Number.parse(
                    tstr.match('PARAMETER\\["elt_0_0",\\s*([-+]?\\d*\\.?\\d+)').get(1)
                ),
                ee.Number.parse(
                    tstr.match('PARAMETER\\["elt_0_2",\\s*([-+]?\\d*\\.?\\d+)').get(1)
                ),
                ee.Number.parse(
                    tstr.match('PARAMETER\\["elt_1_1",\\s*([-+]?\\d*\\.?\\d+)').get(1)
                ),
                ee.Number.parse(
                    tstr.match('PARAMETER\\["elt_1_2",\\s*([-+]?\\d*\\.?\\d+)').get(1)
                ),
            ]
        )
        feat = ee.Feature(
            None,
            {
                "xScale": transform_lst.get(0),
                "xMin": transform_lst.get(1),
                "yScale": transform_lst.get(2),
                "yMax": transform_lst.get(3),
                "system:index": img.get("system:index"),
                "band": "B2",
            },
        )

        return feat

    return ee.FeatureCollection(imgc.map(_img_to_feature))


# def get_grid_and_bounds_for_imgc(
#     imgc: ee.ImageCollection, scale: int
# ) -> Tuple[List[float], dict]:
#     # map over imgc; get transform and bounds
#     #  - img.select(0).projection().transform()
#     fc_transform = imagecollection_to_transform_fc(imgc)

#     # get min xmin and max ymax across all images in the collection
#     min_xmin = fc_transform.aggregate_min("xMin").getInfo()
#     max_xmin = fc_transform.aggregate_max("xMin").getInfo()

#     min_ymax = fc_transform.aggregate_min("yMax").getInfo()
#     max_ymax = fc_transform.aggregate_max("yMax").getInfo()

#     tile_width_meters = 109800  # Sentinel-2 tile width in meters (tile dimensions [10980, 10980] at 10m resolution)

#     xmin, xmax = min_xmin, max_xmin + tile_width_meters
#     ymin, ymax = min_ymax - tile_width_meters, max_ymax

#     bounds_in_utm = [xmin, ymin, xmax, ymax]

#     total_shape_2d = (
#         math.ceil(abs(xmax - xmin) / scale),
#         math.ceil(abs(ymax - ymin) / scale),
#     )

#     # Convert to xarray
#     grid_params = helpers.extract_grid_params(imgc.select("B2"))

#     new_origin = (xmin, ymax)
#     new_crs_transform = (
#         scale,
#         0,
#         new_origin[0],
#         0,
#         -scale,
#         new_origin[1],
#     )
#     grid_params["crs_transform"] = new_crs_transform
#     grid_params["shape_2d"] = total_shape_2d

#     return bounds_in_utm, grid_params


def get_grid_and_bounds_for_mgrs(
    mgrs_tile: str, scale: int
) -> Tuple[List[float], dict]:
    # load from precomputed results
    reference_img = (
        ee.ImageCollection(
            "projects/ee-speckerfelix/assets/open-earth/laie_predictions-mlp_20m_v03"
        )
        .filter(ee.Filter.eq("mgrs_tile", mgrs_tile))
        .first()
    )
    grid_params = helpers.extract_grid_params(reference_img.select("laie_mean"))

    # change scale in transform to match input scale
    new_crs_transform = (
        scale,
        0,
        grid_params["crs_transform"][2],
        0,
        -scale,
        grid_params["crs_transform"][5],
    )
    new_shape_2d = (
        math.ceil(grid_params["shape_2d"][0] * grid_params["crs_transform"][0] / scale),
        math.ceil(
            grid_params["shape_2d"][1] * abs(grid_params["crs_transform"][4]) / scale
        ),
    )
    grid_params["crs_transform"] = new_crs_transform
    grid_params["shape_2d"] = new_shape_2d

    return grid_params


def load_s2_data_for_seed(
    mgrs_tile: str, seed: int, sample_s2_prop: float, sample_lut_prop: float, scale: int
) -> Tuple[xr.Dataset, StandardScaler, NearestNeighbors]:
    """
    Load and process S2 data for a single seed.
    Returns normalized dataset, fitted scaler, and KNN model.
    """
    # logger.info(f"Processing MGRS tile {mgrs_tile}, trait {trait}, seed {seed}")

    # Load LUT data and fit scaler
    lut_spectra = load_lut_for_trait(
        trait=["laie", "fapar", "fcover"],
        ensemble_size=5,
        sample_lut_prop=sample_lut_prop,
        sample_lut_seed=seed,
    )
    standard_scaler = StandardScaler()
    lut_spectra_transformed = standard_scaler.fit_transform(lut_spectra[BAND_NAMES])

    # Load S2 indices and sample with seed
    mgrs_s2_indices = load_s2_imgc_mgrs(mgrs_tile)
    sampled_s2_indices = random_sample_s2_indices(
        mgrs_s2_indices, sample_prop=sample_s2_prop, seed=seed
    )

    if len(sampled_s2_indices) == 0:
        logger.warning(f"No S2 indices found for MGRS tile {mgrs_tile}, seed {seed}")
        return None, None, None

    # Create Earth Engine ImageCollection with seed
    s2_all = ee.ImageCollection(
        (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filter(ee.Filter.inList("system:index", mgrs_s2_indices))
            .select(BAND_NAMES)
        ).randomColumn("random", seed=seed)
    )

    # create grid and bounding box in utm zone from the input imgc-collection
    # bbox_utm, grid_params = get_grid_and_bounds_for_imgc(s2_all, scale)
    grid_params = get_grid_and_bounds_for_mgrs(mgrs_tile, scale)

    s2_sampled = s2_all.filter(ee.Filter.inList("system:index", sampled_s2_indices))

    # Apply cloud mask
    s2_cloudfree = apply_cloudScorePlus_mask(
        s2_sampled, cs_band="cs", cs_threshold=0.65
    ).select(BAND_NAMES)

    # Create mosaic with seed-based sorting
    s2_mosaic = (
        s2_cloudfree.sort("random").mosaic().divide(10000)
    )  # scale to reflectance

    # now we also reproduce masking:
    s2_mosaic = _apply_water_and_rock_mask(s2_mosaic)

    # Apply standard scaling
    ee_standard_scaler = eeStandardScaler(standard_scaler, feature_names=BAND_NAMES)
    s2_normalized = ee_standard_scaler.transform_image(s2_mosaic)

    ds = xr.open_dataset(
        s2_normalized,
        engine="ee",
        **grid_params,
    )

    # Drop time dimension if exists
    if "time" in ds.dims:
        ds = ds.isel(time=0)

    # Add seed as a coordinate
    ds = ds.assign_coords(seed=seed)

    # Create KNN model
    knn_model = NearestNeighbors(
        n_neighbors=N_NEAREST_NEIGHBORS, algorithm="kd_tree", metric="manhattan"
    ).fit(lut_spectra_transformed)

    return ds, standard_scaler, knn_model


def compute_knn_distances_for_dataset(
    ds: xr.Dataset, knn_model: NearestNeighbors
) -> np.ndarray:
    """
    Compute KNN distances for a single dataset.
    Returns 2D array of mean distances.
    """
    # Stack bands into one DataArray: (band, X, Y)
    arr = ds[BAND_NAMES].to_array(dim="band")

    # Move band axis to the end: (X, Y, band)
    arr_np = np.moveaxis(arr.values, 0, -1)

    # Remember spatial shape
    nx, ny, nbands = arr_np.shape

    # Reshape to 2D: (n_pixels, n_bands)
    X_pixels = arr_np.reshape(-1, nbands)

    # Mask invalid pixels
    valid_mask = np.all(np.isfinite(X_pixels), axis=1)

    # Prepare output arrays
    distances_out = np.full((X_pixels.shape[0]), np.nan, dtype=np.float32)

    if np.any(valid_mask):
        # Predict only for valid pixels
        distances, indices = knn_model.kneighbors(X_pixels[valid_mask])

        # Get mean distance to neighbors for each pixel
        mean_distances = distances.mean(axis=1)
        distances_out[valid_mask] = mean_distances

    # Reshape back to raster
    distances_img = distances_out.reshape(nx, ny)

    return distances_img


def process_mgrs_tile(
    mgrs_tile: str,
    seeds: List[int],
    sample_s2_prop: float,
    sample_lut_prop: float,
    scale: int,
    save_as_uint8: bool,
    output_dir: str,
) -> None:
    """
    Process a single MGRS tile with multiple seeds and traits.
    Write each seed individually to disk.
    """

    logger.info(f"Processing MGRS tile {mgrs_tile}")

    for seed in seeds:
        logger.debug(f"Processing seed {seed} for MGRS tile {mgrs_tile}")
        ds, scaler, knn_model = load_s2_data_for_seed(
            mgrs_tile, seed, sample_s2_prop, sample_lut_prop, scale
        )

        if ds is not None:
            # Compute distances for this seed
            distances_img = compute_knn_distances_for_dataset(ds, knn_model)

            # Create DataArray with the distances
            distances_da = xr.DataArray(
                distances_img,
                dims=["y", "x"],
                coords={"x": ds.x, "y": ds.y},
                attrs=ds.attrs,
            )

            # Set spatial domains / crs and transform
            distances_da.rio.write_crs(ds.rio.crs, inplace=True)

            if save_as_uint8:
                # scale distances to 0-250 for uint8 storage / reserve 255 for nodata
                scale_factor = 250 / MAX_RESCALE_DISTANCE
                scaled_distances = (
                    (distances_da * scale_factor).fillna(255).astype(np.uint8)
                )
                scaled_distances.rio.write_nodata(255, inplace=True)
            else:
                scale_factor = 1.0
                scaled_distances = distances_da

            # Define output path with seed number
            output_path = (
                Path(output_dir)
                / f"knn-distance_all-traits_{mgrs_tile}_seed-{seed}.tif"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # write using rasterio to ensure scale and nodata are correctly set in metadata
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=scaled_distances.shape[0],
                width=scaled_distances.shape[1],
                count=1,
                dtype=np.uint8 if save_as_uint8 else np.float32,
                crs=ds.rio.crs,
                transform=scaled_distances.rio.transform(),
                nodata=255 if save_as_uint8 else np.nan,
                compress="deflate",
                tiled=True,
            ) as dst:
                dst.write(scaled_distances.values, 1)
                dst.scales = [1 / scale_factor]
                dst.offsets = [0]


def main():
    """
    Main function to run generalized analysis across multiple MGRS tiles and seeds.
    """
    logger.info("Starting generalized domain distance analysis")

    # Configuration
    seeds = [1, 2, 3, 4, 5]  # Multiple seeds for robustness
    # seeds = [42]
    sample_s2_prop = 0.5  # Proportion of S2 images to sample
    sample_lut_prop = 0.25  # Proportion of LUT samples to use
    scale = 100  # Resolution in meters
    parallel = True  # Whether to run in parallel
    save_as_uint8 = True  # Whether to save distances as uint8 (scaled) or float32

    # Get all MGRS tiles
    # mgrs_tiles = get_mgrs_tiles()
    mgrs_tiles = get_mgrs_from_ref()

    # get already processed mgrs tiles to skip - check if ALL seeds exist for each tile
    output_dir = Path("analysis/domain_representativeness_experiment/test-results")
    output_dir = Path(
        "analysis/domain_representativeness_experiment", f"ood-results-{scale}m"
    )
    if not output_dir.exists():
        logger.info(f"Output directory {output_dir} does not exist yet. Creating it.")
        output_dir.mkdir(parents=True, exist_ok=True)

    existing_mgrs_tiles = set()
    for mgrs_tile in mgrs_tiles:
        all_seeds_exist = True
        for seed in seeds:
            expected_file = (
                output_dir / f"knn-distance_all-traits_{mgrs_tile}_seed-{seed}.tif"
            )
            if not expected_file.exists():
                all_seeds_exist = False
                break
        if all_seeds_exist:
            existing_mgrs_tiles.add(mgrs_tile)

    mgrs_tiles = [tile for tile in mgrs_tiles if tile not in existing_mgrs_tiles]
    logger.info(f"Skipping {len(existing_mgrs_tiles)} already processed MGRS tiles")

    # For testing, limit to a subset (remove this for full run)
    # test_tiles = [
    #     "31T",
    #     "32T",
    #     "33T",
    #     "34T",
    #     "31U",
    #     "32U",
    #     "33U",
    #     "34U",
    # ]  # Limit for testing
    # test_tiles = ["28S"]
    # mgrs_tiles = [tile for tile in mgrs_tiles if tile in test_tiles]

    logger.info(f"Processing {len(mgrs_tiles)} MGRS tiles with {len(seeds)} seeds each")
    logger.info(f"MGRS tiles: {mgrs_tiles}")

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            params = {
                "seeds": seeds,
                "sample_s2_prop": sample_s2_prop,
                "sample_lut_prop": sample_lut_prop,
                "scale": scale,
                "save_as_uint8": save_as_uint8,
                "output_dir": output_dir,
            }
            future_to_tile = {
                executor.submit(process_mgrs_tile, mgrs_tile, **params): mgrs_tile
                for mgrs_tile in mgrs_tiles
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_tile),
                total=len(future_to_tile),
            ):
                mgrs_tile = future_to_tile[future]
                try:
                    future.result()  # If the task raised an exception, this will raise it here
                    logger.info(f"Finished tile {mgrs_tile}")
                except Exception as e:
                    logger.error(f"Error exporting MGRS tile {mgrs_tile}: {e}")
    else:
        for mgrs_tile in tqdm(mgrs_tiles, desc="Processing MGRS tiles"):
            # try:
            process_mgrs_tile(
                mgrs_tile=mgrs_tile,
                seeds=seeds,
                sample_s2_prop=sample_s2_prop,
                sample_lut_prop=sample_lut_prop,
                scale=scale,
                save_as_uint8=save_as_uint8,
                output_dir=output_dir,
            )
        # except Exception as e:
        #     logger.error(f"Error processing MGRS tile {mgrs_tile}: {e}")
        #     continue

    # Summary
    logger.info("Analysis completed!")
    # logger.info(f"Successfully processed {len(all_results)} MGRS tiles")

    # for mgrs_tile, results in all_results.items():
    #     logger.info(f"  {mgrs_tile}: {list(results.keys())}")


if __name__ == "__main__":
    # Initialize Earth Engine
    ee.Initialize(
        project="ee-speckerfelix",
        opt_url="https://earthengine-highvolume.googleapis.com",
    )

    # Run main analysis
    main()
