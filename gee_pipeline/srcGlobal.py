import concurrent.futures
import os
from datetime import datetime
from functools import reduce

import ee
import numpy as np
import pandas as pd
from loguru import logger

from config.config import get_config
from gee_pipeline.utilsAngles import add_angles_from_metadata_to_bands
from gee_pipeline.utilsCloudfree import apply_cloudScorePlus_mask
from gee_pipeline.utilsPhenology import get_start_end_date_phenology_for_ecoregion
from gee_pipeline.utilsPredict import (
    add_random_ensemble_assignment,
    collapse_to_mean_and_stddev_multi_trait,
    eePipelinePredictMap,
)
from gee_pipeline.utilsTiles import get_epsg_code_from_mgrs, get_s2_indices_filtered
from train_pipeline.finalTraining import load_model_ensemble

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def export_ecoregion_per_mgrs_tile(
    eco_id: int | list[int],
) -> None:
    version = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["VERSION"]
    year = int(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["YEAR"])
    output_resolution = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["OUTPUT_RESOLUTION"]

    logger.info(f"Exporting ecoregion: {eco_id}")
    # this is used for initial filtering of the imagecollection in GEE, but not for export
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

    if isinstance(eco_id, int):
        geometry = ecoregions.filter(ee.Filter.eq("ECO_ID", eco_id)).first().geometry()
    elif isinstance(eco_id, list):
        # merge geometries of multiple ecoregions
        geometry = (
            ecoregions.filter(ee.Filter.inList("ECO_ID", eco_id))
            .union()
            .first()
            .geometry()
        )
    else:
        raise ValueError("eco_id must be int or list[int]")

    bounding_box_geometry = geometry.bounds()

    # for export use rasterized ecoregion product instead.
    resolve_ecoregions = ee.Image(
        "projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec"
    ).select("Resolve_Ecoregion")
    if isinstance(eco_id, int):
        export_mask = resolve_ecoregions.eq(eco_id)
    elif isinstance(eco_id, list):
        export_mask = resolve_ecoregions.eq(eco_id[0])
        for eco in eco_id[1:]:
            export_mask = export_mask.Or(resolve_ecoregions.eq(eco))
    else:
        raise ValueError("eco_id must be int or list[int]")

    if isinstance(eco_id, int):
        start_date, end_date, total_days = get_start_end_date_phenology_for_ecoregion(
            eco_id, year
        )
        start_date = ee.Date(start_date)
        end_date = ee.Date(end_date)
    elif isinstance(eco_id, list):
        results = {}
        for eco in eco_id:
            results[eco] = get_start_end_date_phenology_for_ecoregion(eco, year)

        # assert all results are the same
        if len(set(results.values())) != 1:
            logger.warning(
                "Phenology dates are not the same for all ecoregions, Average start and end dates will be used"
            )
            start_dates = [
                datetime.strptime(results[eco][0], "%Y-%m-%d").timestamp()
                for eco in eco_id
            ]
            end_dates = [
                datetime.strptime(results[eco][1], "%Y-%m-%d").timestamp()
                for eco in eco_id
            ]

            average_start_date = np.mean(start_dates)
            average_end_date = np.mean(end_dates)

            start_date = datetime.fromtimestamp(average_start_date).strftime("%Y-%m-%d")
            end_date = datetime.fromtimestamp(average_end_date).strftime("%Y-%m-%d")
            start_date = ee.Date(start_date)
            end_date = ee.Date(end_date)

        else:
            # assert len(set(results.values())) == 1
            start_date, end_date, total_days = results[eco_id[0]]
            start_date = ee.Date(start_date)
            end_date = ee.Date(end_date)
    else:
        raise ValueError("eco_id must be int or list[int]")

    # save s2_indices_filtered for later use
    if isinstance(eco_id, int):
        s2_indices_filename = f"s2-indices_{year}_eco-{eco_id}_{version}.txt"
    elif isinstance(eco_id, list):
        eco_ids_string = "-".join(map(str, eco_id))
        s2_indices_filename = f"s2-indices_{year}_eco-{eco_ids_string}_{version}.txt"
    if os.path.exists(
        os.path.join(
            "data", "gee_pipeline", "outputs", "s2_indices", s2_indices_filename
        )
    ):
        logger.debug(f"Loading s2_indices_filtered from file: {s2_indices_filename}")
        with open(
            os.path.join(
                "data", "gee_pipeline", "outputs", "s2_indices", s2_indices_filename
            ),
            "r",
        ) as f:
            s2_indices_filtered = f.read().splitlines()
    else:
        is_full_year = True if total_days >= 360 else False
        s2_indices_filtered = get_s2_indices_filtered(
            ecoregion_geometry=geometry,
            start_date=start_date,
            end_date=end_date,
            is_full_year=is_full_year,
        )
        logger.debug(f"Saving s2_indices_filtered to file: {s2_indices_filename}")
        # save s2_indices_filtered for later use
        with open(
            os.path.join(
                "data", "gee_pipeline", "outputs", "s2_indices", s2_indices_filename
            ),
            "w",
        ) as f:
            for item in s2_indices_filtered:
                f.write("%s\n" % item)

    if len(s2_indices_filtered) == 0:
        logger.error(
            f"Sentinel-2 collection empty after filter for ecoregion: {eco_id}"
        )
        return

    mgrs_tiles = set(
        system_index.split("_")[2][0:4] for system_index in s2_indices_filtered
    )

    for mgrs_tile in mgrs_tiles:
        current_sentinel2_indices = [
            system_index
            for system_index in s2_indices_filtered
            if system_index.split("_")[2][0:4] == mgrs_tile
        ]
        imgc = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(
            ee.Filter.inList("system:index", current_sentinel2_indices)
        )

        # determine intersecting output geometry
        output_geometry_bbox = imgc.geometry().bounds()

        # apply cloud mask
        imgc = apply_cloudScorePlus_mask(imgc)

        # Apply the function to each image in the collection
        # imgc = imgc.map(add_random_property)
        imgc = ee.ImageCollection(add_random_ensemble_assignment(imgc))

        # add angles to bands
        imgc = imgc.map(add_angles_from_metadata_to_bands)

        imgc_preds = {}

        for trait in CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAITS"]:
            gee_preds = {}
            models = load_model_ensemble(trait=trait, models=["mlp"])
            for i, (model_name, model) in enumerate(models.items()):
                imgc_i = imgc.filter(ee.Filter.eq("random_ensemble_assignment", i + 1))
                gee_preds[model_name] = eePipelinePredictMap(
                    pipeline=model["pipeline"],
                    imgc=imgc_i,
                    trait=trait,
                    model_config=model["config"],
                    min_max_bands=model["min_max_bands"],
                    min_max_label=model["min_max_label"],
                )

            imgc_preds[trait] = reduce(lambda x, y: x.merge(y), gee_preds.values())

        # link the collections if more than one trait
        if len(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAITS"]) > 1:
            imgc_preds_combined = reduce(
                lambda x, y: x.combine(y), list(imgc_preds.values())
            )
        else:
            imgc_preds_combined = imgc_preds[trait]

        # explicitly cast toFloat
        imgc_preds_combined = imgc_preds_combined.map(lambda img: img.toFloat())

        # collapse to mean and stddev
        output_image = collapse_to_mean_and_stddev_multi_trait(imgc_preds_combined)

        # mask permament water bodies :80: permanent water bodies at 10 meter resolution
        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v100").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

        # Set export parameters
        year_start_string = str(year) + "0101"
        year_end_string = str(year) + "1231"
        epsg_code = get_epsg_code_from_mgrs(mgrs_tile)
        epsg_code_gee = f"EPSG:{epsg_code}"
        epsg_string = f"epsg-{epsg_code}"
        if len(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAITS"]) == 1:
            traits_string = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAITS"][0]
        else:
            traits_string = "-".join(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAITS"])
        if isinstance(eco_id, int):
            system_index = f"{traits_string}_rtm-mlp_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_eco-{eco_id}-{mgrs_tile}_{epsg_string}_{version}"
        elif isinstance(eco_id, list):
            eco_ids_string = "-".join(map(str, eco_id))
            system_index = f"{traits_string}_rtm-mlp_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_eco-{eco_ids_string}-{mgrs_tile}_{epsg_string}_{version}"

        output_image = (
            output_image.set(
                "system:time_start", ee.Date.fromYMD(int(year), 1, 1).millis()
            )
            .set("system:time_end", ee.Date.fromYMD(int(year), 12, 31).millis())
            .set("pheno_start", ee.Date(start_date).millis())
            .set("pheno_end", ee.Date(end_date).millis())
            .set("year", year)
            .set("version", version)
            .set("ecoregion_id", eco_id)
            .set("system:index", system_index)
            .set("mgrs_tile", mgrs_tile)
        )

        # Export the image
        imgc_folder = (
            CONFIG_GEE_PIPELINE["GEE_FOLDERS"]["ASSET_FOLDER"]
            + f"/{traits_string}_predictions-mlp_{output_resolution}m_{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['VERSION']}/"
        )

        task = ee.batch.Export.image.toAsset(
            image=output_image.updateMask(export_mask),
            description=system_index,
            crs=epsg_code_gee,
            assetId=imgc_folder + system_index,
            region=output_geometry_bbox,
            scale=output_resolution,
            maxPixels=1e11,
        )
        task.start()


def global_export_concurrent():
    # export all ecoregions
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    all_ecoregions = ecoregions.aggregate_array("ECO_ID").getInfo()

    ecoregions_to_exclude = pd.read_csv(
        os.path.join(
            "data", "phenology_pipeline", "outputs", "ecoregions_to_exclude_all.csv"
        )
    )
    ecoregions_simplifier = get_config("ecoregions_simple")
    ecoregions_in_simplifier = set(
        [item for row in ecoregions_simplifier["same_pheno"] for item in row]
        + [item for row in ecoregions_simplifier["close_pheno"] for item in row]
    )

    ecoregions_process_single_list = list(
        (
            set(all_ecoregions)
            - set(ecoregions_to_exclude["ECO_ID"])
            - set(ecoregions_in_simplifier)
        )
    )

    ecoregions_process_multi_list = [
        *ecoregions_simplifier["same_pheno"],
        *ecoregions_simplifier["close_pheno"],
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(export_ecoregion_per_mgrs_tile, eco_id)
            for eco_id in [
                *ecoregions_process_single_list,
                *ecoregions_process_multi_list,
            ]
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # If the task raised an exception, this will raise it here
            except Exception as e:
                logger.error(f"Error exporting ecoregion: {e}")

    logger.info("All ecoregions export tasks started")


if __name__ == "__main__":
    global_export_concurrent()
