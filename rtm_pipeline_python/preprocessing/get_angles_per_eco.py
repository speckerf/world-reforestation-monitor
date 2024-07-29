import math
import os

import ee
import pandas as pd
import tqdm
from loguru import logger

from validation_pipeline.utils import load_ecoregion_shapefile

ee.Initialize(project="ee-speckerfelix")


def extract_angles_from_s2_tile(img: ee.Image) -> ee.Feature:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    def get_safe_number(prop):
        value = img.getNumber(prop)
        return ee.Algorithms.If(ee.Algorithms.IsEqual(value, None), ee.Number(0), value)

    # Extract the solar azimuth and zenith angles from metadata.
    solar_azimuth = get_safe_number("MEAN_SOLAR_AZIMUTH_ANGLE")
    solar_zenith = get_safe_number("MEAN_SOLAR_ZENITH_ANGLE")

    # Calculate the mean view azimuth angle for the specified bands.
    view_azimuth_list = [
        get_safe_number(f"MEAN_INCIDENCE_AZIMUTH_ANGLE_{b}") for b in bands
    ]
    view_azimuth = ee.Array(view_azimuth_list).reduce(ee.Reducer.mean(), [0]).get([0])

    # Calculate the mean view zenith angle for the specified bands.
    view_zenith_list = [
        get_safe_number(f"MEAN_INCIDENCE_ZENITH_ANGLE_{b}") for b in bands
    ]
    view_zenith = ee.Array(view_zenith_list).reduce(ee.Reducer.mean(), [0]).get([0])

    # Create a feature with the properties
    return ee.Feature(
        None,
        {
            "solar_azimuth": solar_azimuth,
            "solar_zenith": solar_zenith,
            "view_azimuth": view_azimuth,
            "view_zenith": view_zenith,
        },
    )


def get_s2_angles_per_ecoregion(
    region: ee.Geometry, start_date: str, end_date: str, eco_id: int
) -> None:
    s2_reflectances_fc = ee.ImageCollection(
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(["B2"])
        .randomColumn("random", seed=42)
    )

    # if imagecollection is larger than 10000, take the first 10000, otherwise take the whole collection
    s2_reflectances_fc = s2_reflectances_fc.limit(10000, "random", False).map(
        lambda img: extract_angles_from_s2_tile(img)
    )

    # filter out features with null values
    s2_reflectances_fc = s2_reflectances_fc.filter(
        ee.Filter.notNull(
            ["solar_azimuth", "solar_zenith", "view_azimuth", "view_zenith"]
        )
    )

    cloud_storage_bucket = "felixspecker"
    # export the samples
    task = ee.batch.Export.table.toCloudStorage(
        collection=s2_reflectances_fc,
        description=f"s2_angles_eco_{eco_id}",
        bucket=cloud_storage_bucket,
        fileNamePrefix=f"open-earth/s2_reflectances/angles_ecoregion_level/s2_angles_eco_{eco_id}",
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
    pheno_df["start_season_str"] = pheno_df["start_season"].dt.strftime("%Y-%m-%d")
    pheno_df["end_season_str"] = pheno_df["end_season"].dt.strftime("%Y-%m-%d")

    # explicit cast to int requires: numpy.int64 causes error in earth engine
    valid_ecoregions = [int(a) for a in pheno_df["ECO_ID"].unique()]

    for eco_id in tqdm.tqdm(valid_ecoregions):

        logger.info(f"Exporting samples for ecoregion {eco_id}")
        pheno_start = pheno_df.loc[
            pheno_df["ECO_ID"] == eco_id, "start_season_str"
        ].values[0]
        pheno_end = pheno_df.loc[pheno_df["ECO_ID"] == eco_id, "end_season_str"].values[
            0
        ]

        ecoregion_geom = ecoregions.filter(ee.Filter.eq("ECO_ID", eco_id)).geometry()
        get_s2_angles_per_ecoregion(
            region=ecoregion_geom,
            start_date=pheno_start,
            end_date=pheno_end,
            eco_id=eco_id,
        )


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
            "data/rtm_pipeline/input/s2_reflectances/angles_ecoregion_level"
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
    # main()
    _test_all_ecoregions_present()
