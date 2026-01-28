import concurrent.futures
import os
import time
from functools import reduce

import ee
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.config import get_config
from gee_pipeline.utilsAngles import add_angles_from_metadata_to_bands
from gee_pipeline.utilsCloudfree import apply_cloudScorePlus_mask
from gee_pipeline.utilsPredict import eeEnsemblePredictSingleImg, scale_and_cast_to_int
from gee_pipeline.utilsTiles import (
    add_group,
    get_epsg_code_from_mgrs,
    get_s2_indices_filtered,
)
from train_pipeline.finalTraining import load_model_ensemble

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")

# Comment out the following lines to use the default GEE credentials
service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")
# ee.Initialize()


def export_mgrs_tile(mgrs_tile: str) -> None:
    model_version = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["MODEL_VERSION"]
    export_version = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["EXPORT_VERSION"]
    year = int(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["YEAR"])
    output_resolution = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["OUTPUT_RESOLUTION"]
    variable = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["VARIABLE"]

    logger.info(f"Exporting mgrs_tile: {mgrs_tile}")
    start_date = ee.Date(f"{year}-01-01")
    end_date = ee.Date(f"{year}-12-31")

    # list all sentinel-2 tiles in this mgrs tile
    all_mgrs_tiles = pd.read_csv(
        os.path.join(
            "data",
            "gee_pipeline",
            "outputs",
            "mgrs_tiles",
            "mgrs_tiles_all_land_ecoregions.csv",
        )
    )
    current_mgrs_tiles = list(
        set(
            all_mgrs_tiles[all_mgrs_tiles["mgrs_tile_3"] == mgrs_tile][
                "mgrs_tile"
            ].tolist()
        )
    )

    # save s2_indices_filtered for later use
    s2_indices_filename = (
        f"s2-indices_{year}_mgrs-tile-{mgrs_tile}_{export_version}.txt"
    )
    if os.path.exists(
        os.path.join(
            "data",
            "gee_pipeline",
            "outputs",
            "s2_indices_per_mgrs_tile",
            s2_indices_filename,
        )
    ):
        logger.debug(f"Loading s2_indices_filtered from file: {s2_indices_filename}")
        with open(
            os.path.join(
                "data",
                "gee_pipeline",
                "outputs",
                "s2_indices_per_mgrs_tile",
                s2_indices_filename,
            ),
            "r",
        ) as f:
            s2_indices_filtered = f.read().splitlines()
    else:
        s2_indices_filtered = get_s2_indices_filtered(
            mgrs_tiles=current_mgrs_tiles, start_date=start_date, end_date=end_date
        )
        logger.debug(f"Saving s2_indices_filtered to file: {s2_indices_filename}")
        # save s2_indices_filtered for later use
        with open(
            os.path.join(
                "data",
                "gee_pipeline",
                "outputs",
                "s2_indices_per_mgrs_tile",
                s2_indices_filename,
            ),
            "w",
        ) as f:
            for item in s2_indices_filtered:
                f.write("%s\n" % item)

    if len(s2_indices_filtered) == 0:
        logger.error(
            f"Sentinel-2 collection empty after filter for mgrs_tile: {mgrs_tile}"
        )
        return

    imgc = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(
        ee.Filter.inList("system:index", s2_indices_filtered)
    )

    # determine intersecting output geometry
    output_geometry_bbox = imgc.geometry().bounds()

    # apply cloud mask
    imgc = apply_cloudScorePlus_mask(imgc)

    imgc = imgc.map(add_group)

    # add angles to bands
    imgc = imgc.map(add_angles_from_metadata_to_bands)

    models, (uncertainty_calibration_model, uncertainty_calibration_table) = (
        load_model_ensemble(trait=variable)
    )

    # # map over imagecollection
    imgc_preds = imgc.map(
        lambda img: eeEnsemblePredictSingleImg(
            ensemble=models,
            img=img,
            calibrate_uncertainty=True,
            uncertainty_calibration_table=uncertainty_calibration_table,
        )
    )

    # reduce to mean / stdDev_across-images / stdDev_within-images per group
    preds_mean = imgc_preds.select(f"{variable}_mean").mean().rename(f"{variable}_mean")
    preds_count = (
        imgc_preds.select(f"{variable}_mean")
        .reduce(ee.Reducer.count())
        .rename(f"{variable}_count")
    )
    preds_stdDev_within = (
        imgc_preds.select(f"{variable}_stdDev")
        .mean()
        .rename(f"{variable}_stdDev_within")
    )
    preds_stdDev_across = (
        imgc_preds.select(f"{variable}_mean")
        .reduce(ee.Reducer.sampleStdDev())
        .unmask(0)
        .updateMask(preds_stdDev_within.mask())
        .rename(f"{variable}_stdDev_across")
    )
    preds_stdDev_total = (
        preds_stdDev_across.pow(2).add(preds_stdDev_within.pow(2)).sqrt()
    ).rename(f"{variable}_stdDev")

    output_image = ee.Image(
        [
            preds_mean,
            preds_stdDev_total,
            preds_count,
            preds_stdDev_across,
            preds_stdDev_within,
        ]
    )

    # cast to proper datatypes
    if CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["CAST_TO_INT16"]:
        output_image = scale_and_cast_to_int(output_image)

    # mask permament water bodies :80: permanent water bodies at 10 meter resolution
    water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
    output_image = output_image.updateMask(water_mask_2020.neq(80))

    # mask out all rock and ice pixels: ECO_ID = 0 ("Rock and Ice")
    ecoregions = ee.Image(
        "projects/ee-speckerfelix/assets/open-earth/resolve_ecoregions_2017_rasterized"
    ).select("Resolve_Ecoregion")
    output_image = output_image.updateMask(ecoregions.neq(0))

    # Set export parameters
    year_start_string = str(year) + "0101"
    year_end_string = str(year) + "1231"
    epsg_code = get_epsg_code_from_mgrs(mgrs_tile)
    epsg_code_gee = f"EPSG:{epsg_code}"
    epsg_string = f"epsg-{epsg_code}"

    system_index = f"{variable}_rtm-mlp-{model_version}_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_T{mgrs_tile}_{epsg_string}_{export_version}"

    output_image = (
        output_image.set("system:time_start", ee.Date.fromYMD(int(year), 1, 1).millis())
        .set("system:time_end", ee.Date.fromYMD(int(year), 12, 31).millis())
        .set("year", year)
        .set("export_version", export_version)
        .set("model_version", model_version)
        .set("system:index", system_index)
        .set("mgrs_tile", mgrs_tile)
    )

    # Export the image
    imgc_folder = (
        CONFIG_GEE_PIPELINE["GEE_FOLDERS"]["ASSET_FOLDER"]
        + f"/{variable}_predictions-mlp_{output_resolution}m_{export_version}/"
    )

    task = ee.batch.Export.image.toAsset(
        image=output_image,
        description=system_index,
        crs=epsg_code_gee,
        assetId=imgc_folder + system_index,
        region=output_geometry_bbox,
        scale=output_resolution,
        maxPixels=1e11,
    )
    task.start()
    time.sleep(0.1)


def global_export_mgrs_tiles():
    mgrs_tiles = pd.read_csv(
        os.path.join(
            "data",
            "gee_pipeline",
            "outputs",
            "mgrs_tiles",
            "mgrs_tiles_all_land_ecoregions.csv",
        )
    )
    mgrs_tiles_list = list(set(mgrs_tiles["mgrs_tile_3"].tolist()))

    include = ["19F", "19E", "18X", "20F", "42N", "42P", "39M"]
    # mgrs_tiles_list = include
    mgrs_tiles_list = list(set([*mgrs_tiles_list, *include]))
    logger.debug(f"Exporting mgrs_tiles: {mgrs_tiles_list}")

    # mgrs_tiles_list = [ # for figures
    #     "36N",
    #     "10T",
    #     "10S",
    #     "15Q",
    #     "16Q",
    #     "34N",
    #     "55G",
    #     "31T",
    #     "24L",
    #     "10U",
    #     "10U",
    #     "35L",
    #     "32U",
    # ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(export_mgrs_tile, mgrs_tile)
            for mgrs_tile in mgrs_tiles_list
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting mgrs_tile: {e}")

    logger.info("All mgrs_tile export tasks started")


if __name__ == "__main__":
    # wait 1 hour
    # time.sleep(3 * 3600)
    global_export_mgrs_tiles()
