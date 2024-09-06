import datetime
import glob
import json
import os

import ee
import geopandas as gpd
import pandas as pd
from loguru import logger

from validation_pipeline.utils import (
    add_closest_cloudfree_s2_image_reflectances,
    load_ecoregion_shapefile,
)
from validation_pipeline.utils_foliar import merge_foliar_files

service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def foliar_directories() -> list:
    return [
        (
            os.path.join("data", "validation_pipeline", "input", "traits_NEON"),
            os.path.join(
                "data",
                "validation_pipeline",
                "output",
                "foliar",
                "neon_foliar_insitu_merged.csv",
            ),
        )
    ]


def foliar_points():
    logger.info(
        "Getting leaf mass per area and equivalent water thickness points from NEON validation data, and get s2 reflectances for each point."
    )

    dirs = foliar_directories()

    for basedir, output in dirs:
        # Pattern to match the directories
        dir_pattern = f"{basedir}/*basic.20230127T120753Z.RELEASE-2023"
        directories = glob.glob(dir_pattern)
        logger.info(
            f"Merging NEON validation data from folder {basedir} and saving to {output}"
        )

        df = merge_foliar_files(directories)

        df = df.rename(
            columns={
                "decimalLatitude": "lat",
                "decimalLongitude": "lon",
                "collectDate": "date",
            }
        )

        df["system:time_start"] = (
            df["date"].apply(lambda x: pd.Timestamp(x).timestamp() * 1000).astype("int")
        )  # convert date to posix timestamp milliseconds

        # chlorophyll
        df["chlorophyll_a_mg_gdryleaf"] = (
            (df["extractChlAConc"] * df["solventVolume"])
            / (df["freshMass_chl"] * df["dryMassFraction"])
        ) * 0.001  # convert from mg/g to mg/g dry leaf
        df["chlorophyll_b_mg_gdryleaf"] = (
            (df["extractChlBConc"] * df["solventVolume"])
            / (df["freshMass_chl"] * df["dryMassFraction"])
        ) * 0.001  # convert from mg/g to mg/g dry leaf
        df["chlorophyll_a_mg_m2"] = (
            df["chlorophyll_a_mg_gdryleaf"] * df["leafMassPerArea"]
        )  # convert from mg/g dry leaf to mg/m2
        df["chlorophyll_b_mg_m2"] = (
            df["chlorophyll_b_mg_gdryleaf"] * df["leafMassPerArea"]
        )  # convert from mg/g dry leaf to mg/m2
        df["chlorophyll_ab_mg_m2"] = (
            df["chlorophyll_a_mg_m2"] + df["chlorophyll_b_mg_m2"]
        )
        df["chlorophyll_ab_mug_cm2"] = (
            df["chlorophyll_ab_mg_m2"] * 1000 / 10000
        )  # convert from mg/m2 to µg/cm2
        # carotenoid
        df["carotenoid_mg_gdryleaf"] = (
            (df["extractCarotConc"] * df["solventVolume"])
            / (df["freshMass_chl"] * df["dryMassFraction"])
        ) * 0.001
        df["carotenoid_mg_m2"] = df["carotenoid_mg_gdryleaf"] * df["leafMassPerArea"]
        df["carotenoid_mug_cm2"] = df["carotenoid_mg_m2"] * 1000 / 10000
        # lma
        df["leafMassPerArea_g_cm2"] = df["leafMassPerArea"] / 100**2
        # ewt
        df["ewt_cm"] = (df["freshMass_lma"] - df["dryMass"]) / (
            df["leafArea"] / 100
        )  # equivalent water thickness = (fresh mass - dry mass) / leaf area ; [g/cm2] or [cm] by multiplying with density of water: 1 g/cm3

        # Write to disk
        df.to_csv(output, index=False)
        logger.info(f"Data merged and saved to {output}")

        df_subset = df[
            [
                "plotID",
                "lat",
                "lon",
                "system:time_start",
                "chlorophyll_ab_mug_cm2",
                "leafMassPerArea_g_cm2",
                "ewt_cm",
                "carotenoid_mug_cm2",
            ]
        ]

        # convert to geopandas dataframe with point geometry
        gdf = gpd.GeoDataFrame(
            df_subset, geometry=gpd.points_from_xy(df_subset.lon, df_subset.lat)
        )

        # add column ECO_ID from ecoregions gdf
        ecoregions = load_ecoregion_shapefile()
        gdf = gpd.sjoin(
            gdf, ecoregions[["ECO_ID", "geometry"]], how="left", op="within"
        ).drop(columns="index_right")

        gdf_json = gdf.drop(["lat", "lon"], axis=1).to_json()

        fc = ee.FeatureCollection(json.loads(gdf_json))

        fc_reflectances = fc.map(add_closest_cloudfree_s2_image_reflectances)
        fc_reflectances_nonull = fc_reflectances.filter(
            ee.Filter.notNull(["closest_image", "B2"])
        )

        # export to CloudStorage
        output_filename = "open-earth/validation/NEON_foliar_reflectances_with_angles"
        task = ee.batch.Export.table.toCloudStorage(
            collection=fc_reflectances_nonull,
            description="foliar_export_validation_with_angles",
            bucket="felixspecker",
            fileNamePrefix=output_filename,
        )
        task.start()
        logger.info(
            f"Exported NEON validation data with s2 reflectances to {output_filename}"
        )

        placeholder_filename = output.replace("merged", "export")
        with open(
            os.path.join(f"{placeholder_filename}.txt"),
            "w",
        ) as f:
            f.write(
                f"Exported at time: {datetime.datetime.now()} \n See results at GCS: {output_filename}"
            )
    return None


if __name__ == "__main__":
    foliar_points()
