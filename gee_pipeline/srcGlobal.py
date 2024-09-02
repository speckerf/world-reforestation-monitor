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
    collapse_to_mean_and_stddev,
    eePipelinePredictMap,
)
from gee_pipeline.utilsTiles import get_epsg_code_from_mgrs, groupby_mgrs_orbit_pandas
from train_pipeline.finalTraining import load_model_ensemble

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def get_s2_indices_filtered(
    ecoregion_geometry, start_date: ee.Date, end_date: ee.Date
) -> pd.DataFrame:
    # load s2 data
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ecoregion_geometry)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                CONFIG_GEE_PIPELINE["CLOUD_FILTERING"]["CLOUDY_PIXEL_PERCENTAGE"],
            )
        )
        .filter(
            ee.Filter.And(
                ee.Filter.eq("GENERAL_QUALITY", "PASSED"),
                ee.Filter.eq("GEOMETRIC_QUALITY", "PASSED"),
                ee.Filter.gt("system:asset_size", 1000000),
            )
        )
        .select(bands)
    )

    # filter imgc by grouping by mgrs tile and orbit number
    s2_indices_filtered = groupby_mgrs_orbit_pandas(
        imgc,
        center_pheno=True,
        start_pheno=start_date,
        end_pheno=end_date,
    )
    return s2_indices_filtered


def export_ecoregion(
    eco_id: int | list[int],
    year: int,
    output_resolution: int,
) -> None:

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

    s2_indices_filtered = get_s2_indices_filtered(
        ecoregion_geometry=geometry,
        start_date=start_date,
        end_date=end_date,
    )

    if len(s2_indices_filtered) == 0:
        logger.error(
            f"Sentinel-2 collection empty after filter for ecoregion: {eco_id}"
        )
        return

    del imgc
    imgc = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(
        ee.Filter.inList("system:index", s2_indices_filtered)
    )

    # apply cloud mask
    imgc = apply_cloudScorePlus_mask(imgc)

    # Apply the function to each image in the collection
    # imgc = imgc.map(add_random_property)
    imgc = ee.ImageCollection(add_random_ensemble_assignment(imgc))

    # add angles to bands
    imgc = imgc.map(add_angles_from_metadata_to_bands)

    gee_preds = {}
    # load model ensemble
    models = load_model_ensemble(trait=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"])
    for i, (model_name, model) in enumerate(models.items()):
        imgc_i = imgc.filter(ee.Filter.eq("random_ensemble_assignment", i + 1))
        gee_random_forest_model = (
            None if "rf" not in model_name else model["gee_classifier"]
        )
        gee_preds[model_name] = eePipelinePredictMap(
            pipeline=model["pipeline"],
            imgc=imgc_i,
            trait=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"],
            model_config=model["config"],
            gee_random_forest=gee_random_forest_model,
            min_max_bands=model["min_max_bands"],
            min_max_label=model["min_max_label"],
        )

    # create single imagecollection from all imagecollections using reduce
    imgc_preds = reduce(lambda x, y: x.merge(y), gee_preds.values())

    # explicitly cast toFloat
    imgc_preds = imgc_preds.map(lambda img: img.toFloat())

    # collapse to mean and stddev
    output_image = collapse_to_mean_and_stddev(imgc_preds)

    # mask permament water bodies :80: permanent water bodies at 10 meter resolution
    water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v100").first()
    output_image = output_image.updateMask(water_mask_2020.neq(80))

    # Set export parameters
    year_start_string = str(year) + "0101"
    year_end_string = str(year) + "1231"
    epsg_code = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["EXPORT_EPSG"]
    epsg_string = epsg_code.lower().replace(":", "-")
    version = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["VERSION"]
    if isinstance(eco_id, int):
        system_index = f"{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['TRAIT']}_rtm-ensemble_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_eco-{eco_id}_{epsg_string}_{version}"
    elif isinstance(eco_id, list):
        eco_ids_string = "-".join(map(str, eco_id))
        system_index = f"{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['TRAIT']}_rtm-ensemble_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_eco-{eco_ids_string}_{epsg_string}_{version}"

    output_image = (
        output_image.set("system:time_start", ee.Date.fromYMD(int(year), 1, 1).millis())
        .set("system:time_end", ee.Date.fromYMD(int(year), 12, 31).millis())
        .set("pheno_start", ee.Date(start_date).millis())
        .set("pheno_end", ee.Date(end_date).millis())
        .set("year", year)
        .set("version", version)
        .set("ecoregion_id", eco_id)
        .set("trait", CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"])
        .set("system:index", system_index)
    )

    # Export the image
    imgc_folder = (
        CONFIG_GEE_PIPELINE["GEE_FOLDERS"]["ASSET_FOLDER"]
        + f"/{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['TRAIT']}_predictions_{output_resolution}m_{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['VERSION']}/"
    )

    task = ee.batch.Export.image.toAsset(
        image=output_image.updateMask(export_mask),
        description=system_index,
        crs=epsg_code,
        assetId=imgc_folder + system_index,
        region=bounding_box_geometry,
        scale=output_resolution,
        maxPixels=1e11,
    )
    task.start()


def export_ecoregion_per_mgrs_tile(
    eco_id: int | list[int],
    year: int,
    output_resolution: int,
) -> None:

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

    s2_indices_filtered = get_s2_indices_filtered(
        ecoregion_geometry=geometry,
        start_date=start_date,
        end_date=end_date,
    )

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

        # apply cloud mask
        imgc = apply_cloudScorePlus_mask(imgc)

        # Apply the function to each image in the collection
        # imgc = imgc.map(add_random_property)
        imgc = ee.ImageCollection(add_random_ensemble_assignment(imgc))

        # add angles to bands
        imgc = imgc.map(add_angles_from_metadata_to_bands)

        gee_preds = {}
        # load model ensemble
        models = load_model_ensemble(
            trait=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"]
        )
        for i, (model_name, model) in enumerate(models.items()):
            imgc_i = imgc.filter(ee.Filter.eq("random_ensemble_assignment", i + 1))
            gee_random_forest_model = (
                None if "rf" not in model_name else model["gee_classifier"]
            )
            gee_preds[model_name] = eePipelinePredictMap(
                pipeline=model["pipeline"],
                imgc=imgc_i,
                trait=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"],
                model_config=model["config"],
                gee_random_forest=gee_random_forest_model,
                min_max_bands=model["min_max_bands"],
                min_max_label=model["min_max_label"],
            )

        # create single imagecollection from all imagecollections using reduce
        imgc_preds = reduce(lambda x, y: x.merge(y), gee_preds.values())

        # explicitly cast toFloat
        imgc_preds = imgc_preds.map(lambda img: img.toFloat())

        # collapse to mean and stddev
        output_image = collapse_to_mean_and_stddev(imgc_preds)

        # mask permament water bodies :80: permanent water bodies at 10 meter resolution
        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v100").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

        # Set export parameters
        year_start_string = str(year) + "0101"
        year_end_string = str(year) + "1231"
        epsg_code = get_epsg_code_from_mgrs(mgrs_tile)
        epsg_code_gee = f"EPSG:{epsg_code}"
        epsg_string = f"epsg-{epsg_code}"
        version = CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["VERSION"]
        if isinstance(eco_id, int):
            system_index = f"{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['TRAIT']}_rtm-ensemble_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_eco-{eco_id}-{mgrs_tile}_{epsg_string}_{version}"
        elif isinstance(eco_id, list):
            eco_ids_string = "-".join(map(str, eco_id))
            system_index = f"{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['TRAIT']}_rtm-ensemble_mean-std-n_{output_resolution}m_s_{year_start_string}_{year_end_string}_eco-{eco_ids_string}-{mgrs_tile}_{epsg_string}_{version}"

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
            .set("trait", CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"])
            .set("system:index", system_index)
            .set("mgrs_tile", mgrs_tile)
        )

        # Export the image
        imgc_folder = (
            CONFIG_GEE_PIPELINE["GEE_FOLDERS"]["ASSET_FOLDER"]
            + f"/{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['TRAIT']}_predictions_{output_resolution}m_{CONFIG_GEE_PIPELINE['PIPELINE_PARAMS']['VERSION']}/"
        )

        task = ee.batch.Export.image.toAsset(
            image=output_image.updateMask(export_mask),
            description=system_index,
            crs=epsg_code_gee,
            assetId=imgc_folder + system_index,
            region=bounding_box_geometry,
            scale=output_resolution,
            maxPixels=1e11,
        )
        task.start()


def export_helper():

    # test export allecoregions within this bounding box
    bbox = ee.Geometry.Polygon(
        [
            [
                [5.798245436836389, 51.698735932570415],
                [5.798245436836389, 44.306898298642395],
                [25.74941731183639, 44.306898298642395],
                [25.74941731183639, 51.698735932570415],
            ]
        ],
        None,
        False,
    )

    # get all ecoregions within bounding box
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    ecoregions_bbox = ecoregions.filterBounds(bbox)
    ecoregions_to_export = ecoregions_bbox.aggregate_array("ECO_ID").getInfo()

    for eco_id in ecoregions_to_export:
        logger.info(f"Exporting ecoregion {eco_id}")

        if eco_id not in [799, 644, 660]:
            continue
        export_ecoregion(
            eco_id=eco_id,
            year=int(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["YEAR"]),
            output_resolution=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"][
                "OUTPUT_RESOLUTION"
            ],
            trait=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"],
        )
    pass


def test_multi_eco():

    # get all ecoregions within bounding box
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    # ecoregions_bbox = ecoregions.filter(ee.Filter.inList("ECO_ID", [146, 164]))
    # ecoregions_to_export = ecoregions_bbox.aggregate_array("ECO_ID").getInfo()

    ecoregions_simplifier = get_config("ecoregions_simple")

    same_pheno = ecoregions_simplifier["same_pheno"]
    close_pheno = ecoregions_simplifier["close_pheno"]

    ecoregions_to_export = [*close_pheno, *same_pheno]

    for eco_combo in ecoregions_to_export:
        logger.info(f"Exporting ecoregions {eco_combo}")

        export_ecoregion(
            eco_id=eco_combo,
            year=int(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["YEAR"]),
            output_resolution=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"][
                "OUTPUT_RESOLUTION"
            ],
            trait=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["TRAIT"],
        )

    pass


def test_export_global(export_per_mgrs_tile: bool = False):
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

    for eco_id in [*ecoregions_process_single_list, *ecoregions_process_multi_list]:
        logger.info(f"Exporting ecoregion {eco_id}")

        if eco_id in list(range(1, 1000)):
            continue

        if export_per_mgrs_tile:
            export_ecoregion_per_mgrs_tile(
                eco_id=eco_id,
                year=int(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["YEAR"]),
                output_resolution=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"][
                    "OUTPUT_RESOLUTION"
                ],
            )
        else:
            export_ecoregion(
                eco_id=eco_id,
                year=int(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["YEAR"]),
                output_resolution=CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"][
                    "OUTPUT_RESOLUTION"
                ],
            )

    pass


if __name__ == "__main__":
    # export_continent()
    # export_helper()
    test_export_global(export_per_mgrs_tile=True)
    # test_multi_eco()
