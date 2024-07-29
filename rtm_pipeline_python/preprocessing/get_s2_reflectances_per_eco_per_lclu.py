import math
import os

import ee
import pandas as pd
import tqdm
from loguru import logger

from validation_pipeline.utils import load_ecoregion_shapefile

ee.Initialize(project="ee-speckerfelix")


def add_angles_from_metadata_to_bands(img: ee.Image) -> ee.Image:
    # Define the bands for which view angles are extracted from metadata.
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # Extract the solar azimuth and zenith angles from metadata.
    solar_azimuth = img.getNumber("MEAN_SOLAR_AZIMUTH_ANGLE")
    solar_zenith = img.getNumber("MEAN_SOLAR_ZENITH_ANGLE")

    # Calculate the mean view azimuth angle for the specified bands.
    view_azimuth = (
        ee.Array([img.getNumber("MEAN_INCIDENCE_AZIMUTH_ANGLE_%s" % b) for b in bands])
        .reduce(ee.Reducer.mean(), [0])
        .get([0])
    )

    # Calculate the mean view zenith angle for the specified bands.
    view_zenith = (
        ee.Array([img.getNumber("MEAN_INCIDENCE_ZENITH_ANGLE_%s" % b) for b in bands])
        .reduce(ee.Reducer.mean(), [0])
        .get([0])
    )

    img = ee.Image.cat(
        [
            img,
            ee.Image.constant(solar_azimuth).toFloat().rename("solar_azimuth"),
            ee.Image.constant(solar_zenith).toFloat().rename("solar_zenith"),
            ee.Image.constant(view_azimuth).toFloat().rename("view_azimuth"),
            ee.Image.constant(view_zenith).toFloat().rename("view_zenith"),
        ]
    )

    return img


def get_scl_label_and_mask(
    start_date: str, end_date: str, region: ee.Geometry
) -> ee.Image:
    #  5  #ffff52 Bare Soils
    #  6  #0000ff Water
    #  11 #52fff9 Snow / Ice
    scl_img = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .select("SCL")
    )
    scl_mode = scl_img.mode().rename("scl_10m_label")
    scl_categories = ee.List([5, 6, 11])
    scl_mask = scl_mode.remap(
        scl_categories, ee.List.repeat(1, scl_categories.size()), 0
    ).rename("scl_10m_mask")
    return {"scl_label": scl_mode, "scl_mask": scl_mask}


def get_dynamic_world_label_and_mask(
    start_date: str, end_date: str, region: ee.Geometry
) -> ee.Image:
    # shift start_date and end_date by 2 years to the future
    start_date = ee.Date(start_date).advance(2, "year").format("YYYY-MM-dd").getInfo()
    end_date = ee.Date(end_date).advance(2, "year").format("YYYY-MM-dd").getInfo()

    # 0 	#419bdf 	water
    # 6 	#c4281b 	built
    # 7 	#a59b8f 	bare
    # 8 	#b39fe1 	snow_and_ice
    dwCol = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start_date, end_date)
        .filterBounds(region.bounds())
    )
    s2Col = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterDate(start_date, end_date)
        .filterBounds(region.bounds())
    )
    s2_dw_merged = s2Col.linkCollection(dwCol, dwCol.first().bandNames())
    dw_mode = (
        ee.ImageCollection(s2_dw_merged)
        .select("label")
        .map(lambda img: img.toUint8())
        .mode()
        .rename("dw_10m_label")
    )
    dw_categories = ee.List([0, 6, 7, 8])
    dw_mask = dw_mode.remap(
        dw_categories, ee.List.repeat(1, dw_categories.size()), 0
    ).rename("dw_10m_mask")
    return {"dw_label": dw_mode, "dw_mask": dw_mask}


def get_proba_label_and_mask() -> ee.Image:
    # 50 #fa0000 Urban / built up. Land covered by buildings and other man-made structures.
    # 60 #b4b4b4 Bare / sparse vegetation. Lands with exposed soil, sand, or rocks and never has more than 10 % vegetated cover during any time of the year.
    # 70 #f0f0f0 Snow and ice. Lands under snow or ice cover throughout the year
    # 80 #0032c8 Permanent water bodies. Lakes, reservoirs, and rivers. Can be either fresh or salt-water bodies
    proba_100m = (
        ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
        .select("discrete_classification")
        .rename("proba_100m_label")
    )
    proba_categories = ee.List([50, 60, 70, 80, 200])
    proba_100m_mask = proba_100m.remap(
        proba_categories, ee.List.repeat(1, proba_categories.size()), 0
    ).rename("proba_100m_mask")

    return {"proba_label": proba_100m, "proba_mask": proba_100m_mask}


def get_esa_worldcover_label_and_mask() -> ee.Image:
    # 50 #fa0000 Built-up
    # 60 #b4b4b4 Bare / sparse vegetation
    # 70 #f0f0f0 Snow and ice
    # 80 #0064c8 Permanent water bodies
    esa_10m = (
        ee.ImageCollection("ESA/WorldCover/v100")
        .filterDate("2020")
        .first()
        .rename("esa_10m_label")
    )
    esa_categories = ee.List([50, 60, 70, 80])
    esa_10m_mask = esa_10m.remap(
        esa_categories, ee.List.repeat(1, esa_categories.size()), 0
    ).rename("esa_10m_mask")
    return {"esa_label": esa_10m, "esa_mask": esa_10m_mask}


def get_s2_reflectances_per_ecoregion(
    region: ee.Geometry, start_date: str, end_date: str
) -> ee.Image:
    """Gets sentinel-2 reflectances for a given region and time period, creates a random mosaic for data point sampling.

    Args:
        region (ee.Geometry): _description_
        start_date (str): _description_
        end_date (str): _description_

    Returns:
        ee.Image: _description_
    """
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    QA_BAND = "cs"
    CLEAR_THRESHOLD = 0.6
    reflectance_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angle_bands = ["solar_azimuth", "solar_zenith", "view_azimuth", "view_zenith"]
    s2_collection = ee.ImageCollection(
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(
            region.bounds()
        )  # use .bounds() to get the bounding box of the region
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
        # filter out all images which don't have the property MEAN_SOLAR_AZIMUTH_ANGLE
        .filter(
            ee.Filter.notNull(
                [
                    "MEAN_SOLAR_AZIMUTH_ANGLE",
                    "MEAN_SOLAR_ZENITH_ANGLE",
                    *[f"MEAN_INCIDENCE_AZIMUTH_ANGLE_{b}" for b in reflectance_bands],
                    *[f"MEAN_INCIDENCE_ZENITH_ANGLE_{b}" for b in reflectance_bands],
                ]
            )
        )
        .randomColumn("random", seed=0)
        .limit(1000, "random", False)
    )

    s2_cldfr = (
        s2_collection.map(lambda img: add_angles_from_metadata_to_bands(img))
        .linkCollection(csPlus, [QA_BAND])
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
        .select([*reflectance_bands, *angle_bands])
    )

    s2_mosaic = s2_cldfr.map(
        lambda img: img.addBands(
            ee.Image.constant(img.getNumber("random")).toFloat().rename("random")
        )
    ).qualityMosaic("random")

    return s2_mosaic


def sample_and_export_s2_reflectances_per_ecoregion(
    region: ee.Geometry,
    eco_id: int,
    pheno_start: str,
    pheno_end: str,
    num_samples: int,
    cloud_storage_bucket: str = "felixspecker",
) -> ee.Feature:
    s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # get the SCL label and mask
    # scl = get_scl_label_and_mask(pheno_start, pheno_end, region)
    # get the dynamic world label and mask
    dw = get_dynamic_world_label_and_mask(pheno_start, pheno_end, region)
    # get the proba label and mask
    proba = get_proba_label_and_mask()
    # get the esa world cover label and mask
    esa = get_esa_worldcover_label_and_mask()

    # lc_labels = ee.Image.cat(
    #     [scl["scl_label"], dw["dw_label"], proba["proba_label"], esa["esa_label"]]
    # ).rename(["scl_label", "dw_label", "proba_label", "esa_label"])

    # lc_masks = ee.Image.cat(
    #     [scl["scl_mask"], dw["dw_mask"], proba["proba_mask"], esa["esa_mask"]]
    # ).rename(["scl_mask", "dw_mask", "proba_mask", "esa_mask"])

    lc_labels = ee.Image.cat(
        [proba["proba_label"], esa["esa_label"], dw["dw_label"]]
    ).rename(["proba_label", "esa_label", "dw_label"])

    lc_masks = ee.Image.cat(
        [proba["proba_mask"], esa["esa_mask"], dw["dw_mask"]]
    ).rename(["proba_mask", "esa_mask", "dw_mask"])

    sum_masks = lc_masks.reduce(ee.Reducer.sum()).rename("sum_masks")

    s2_reflectances = get_s2_reflectances_per_ecoregion(region, pheno_start, pheno_end)

    s2_combined = ee.Image.cat(s2_reflectances, lc_labels, lc_masks, sum_masks)

    # get random points in the ecoregion
    random_points = ee.FeatureCollection.randomPoints(region, num_samples, 0, 1)

    # sample the reflectances
    samples = s2_combined.sampleRegions(
        collection=random_points, scale=10, geometries=True
    )

    # add columns for the ecoregion id
    samples = samples.map(lambda f: f.set("ECO_ID", eco_id))

    # export the samples
    task = ee.batch.Export.table.toCloudStorage(
        collection=samples,
        description=f"s2_reflectances_eco_{eco_id}",
        bucket=cloud_storage_bucket,
        fileNamePrefix=f"open-earth/s2_reflectances/reflectances_angles_ecoregion_level_with_lulc/s2_reflectances_{num_samples}_eco_{eco_id}",
        fileFormat="CSV",
    )
    task.start()


def main():
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    unbounded_geo = ee.Geometry.BBox(-180, -88, 180, 88)

    # load the growing season start and end dates
    pheno_df = pd.read_csv(
        os.path.join(
            "data",
            "gee_pipeline",
            "inputs",
            "phenology",
            "artificial_masked_w_amplitude_singleeco.csv",
        )
    )

    # drop nan values
    pheno_df = pheno_df.dropna(subset=["start_season", "end_season"])

    # convert start_season and end_season to datetime from string %m-%d, set year to 2020
    pheno_df["start_season"] = pd.to_datetime(
        "2020-" + pheno_df["start_season"], format="%Y-%m-%d"
    )
    pheno_df["end_season"] = pd.to_datetime(
        "2020-" + pheno_df["end_season"], format="%Y-%m-%d"
    )

    # if start_season is after end_season, then subtract one year from start_season
    pheno_df.loc[pheno_df["start_season"] > pheno_df["end_season"], "start_season"] = (
        pheno_df.loc[pheno_df["start_season"] > pheno_df["end_season"], "start_season"]
        - pd.DateOffset(years=1)
    )

    # convert start_season and end_season to string format %Y-%m-%d
    pheno_df.loc[:, "start_season_str"] = pheno_df["start_season"].dt.strftime(
        "%Y-%m-%d"
    )
    pheno_df.loc[:, "end_season_str"] = pheno_df["end_season"].dt.strftime("%Y-%m-%d")

    # explicit cast to int requires: numpy.int64 causes error in earth engine
    valid_ecoregions = [int(a) for a in pheno_df["ECO_ID"].unique()]

    for eco_id in tqdm.tqdm(valid_ecoregions):
        # if eco_id not in [53]:
        #     continue
        if eco_id not in [
            384,
            393,
            399,
            39,
            10,
            168,
            41,
            42,
            43,
            53,
            182,
            55,
            440,
            185,
            186,
            320,
            65,
            67,
            837,
            76,
            79,
            81,
            337,
            215,
            88,
            89,
            90,
            347,
            220,
            612,
            230,
            497,
            244,
            757,
        ]:
            continue

        ecoregion_geom = ecoregions.filter(ee.Filter.eq("ECO_ID", eco_id)).geometry()
        logger.info(f"Exporting samples for ecoregion {eco_id}")
        pheno_start = pheno_df.loc[
            pheno_df["ECO_ID"] == eco_id, "start_season_str"
        ].values[0]
        pheno_end = pheno_df.loc[pheno_df["ECO_ID"] == eco_id, "end_season_str"].values[
            0
        ]
        sample_and_export_s2_reflectances_per_ecoregion(
            region=ecoregion_geom,
            eco_id=eco_id,
            pheno_start=pheno_start,
            pheno_end=pheno_end,
            num_samples=10000 / 2,
        )

        # print(f"Exported samples for ecoregion {eco_id}")


def _test_all_ecoregions_present():
    # read csv with ecoregions to be excluded
    eco_ids_excluded = pd.read_csv(
        os.path.join(
            "config",
            "ecoregions_to_exclude_all.csv",
        )
    )["ECO_ID"].values

    ecoregions_gee = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

    # load shapefile with all ecoregions
    ecoregions_shp = load_ecoregion_shapefile()
    ecoregions_shp_ids = list(set(ecoregions_shp["ECO_ID"].values))

    # get list of all ecoregions successfully exported: see repo data/rtm_pipeline/output/s2_reflectances/angles_ecoregion_level
    exported_ecoregions = [
        int(a.split("_")[-1].split(".")[0])
        for a in os.listdir(
            "data/rtm_pipeline/input/s2_reflectances/reflectances_angles_ecoregion_level_with_lulc"
        )
    ]
    exported_ecoregions = list(set(exported_ecoregions))

    ecoids_to_reprocess = list(
        set(ecoregions_shp_ids) - set(exported_ecoregions) - set(eco_ids_excluded)
    )

    if len(ecoids_to_reprocess) > 0:
        logger.info(
            f"Found {len(ecoids_to_reprocess)} ecoregions that need to be reprocessed"
        )
        logger.info(ecoids_to_reprocess)


if __name__ == "__main__":
    main()
    # _test_all_ecoregions_present()
