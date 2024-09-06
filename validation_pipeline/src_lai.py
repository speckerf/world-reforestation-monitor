import datetime
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
from validation_pipeline.utils_lai import merge_lai_files

service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def lai_directories() -> tuple:
    folder_name = "COPERNICUS_GBOV_RM6,7_20240620120826"
    # return [(os.path.join('data', 'validation_pipeline', 'GBOV', 'COPERNICUS_GBOV_RM6,7_20230828153117', 'RM07'), os.path.join('data', 'validation_pipeline', 'GBOV', 'merged_COPERNICUS_GBOV_RM6,7_20230828153117_RM07.csv'))]
    return (
        os.path.join(
            "data",
            "validation_pipeline",
            "input",
            "traits_GBOV",
            folder_name,
            "RM07",
        ),
        os.path.join(
            "data",
            "validation_pipeline",
            "output",
            "lai",
            f"merged_lai_{folder_name}.csv",
        ),
        folder_name,
    )


def lai_points():
    logger.info(
        "Fetching leaf area index points from GBOV validation data, and get s2 reflectances for each point."
    )

    input, output, foldername = lai_directories()

    logger.info(
        f"Merging GBOV validation data from folder {input} and saving to {output}"
    )

    merge_lai_files(input, output)

    df = pd.read_csv(output)
    sample_cols = [
        "system:time_start",
        "uuid",
        "GBOV_ID",
        "PLOT_ID",
        "Lat_IS",
        "Lon_IS",
        "TIME_IS",
    ]
    lai_cols = [
        "LAI_Warren",
        "LAIe_Warren",
        "LAI_Miller",
        "LAIe_Miller",
        "LAI_Warren_err",
        "LAIe_Warren_err",
        "LAI_Miller_err",
        "LAIe_Miller_err",
    ]
    df_subset = df[[*sample_cols, *lai_cols]].rename(
        columns={"Lat_IS": "lat", "Lon_IS": "lon", "TIME_IS": "date"}
    )

    # convert to geopandas dataframe with point geometry
    gdf = gpd.GeoDataFrame(
        df_subset, geometry=gpd.points_from_xy(df_subset.lon, df_subset.lat)
    )

    # add column ECO_ID from ecoregions gdf
    ecoregions = load_ecoregion_shapefile()
    gdf = gpd.sjoin(
        gdf, ecoregions[["ECO_ID", "geometry"]], how="left", op="within"
    ).drop("index_right", axis=1)

    # new column SITE_ID with first part of PLOT_ID (everything before '_')
    gdf["SITE_ID"] = gdf["PLOT_ID"].apply(lambda x: x.split("_")[0])

    # loop over all sites and export reflectances
    for site in gdf["SITE_ID"].unique():
        gdf_site = gdf[gdf["SITE_ID"] == site]
        gdf_json = gdf_site.drop(["lat", "lon"], axis=1).to_json()
        fc = ee.FeatureCollection(json.loads(gdf_json))

        fc_reflectances = fc.map(add_closest_cloudfree_s2_image_reflectances)
        fc_reflectances_nonull = fc_reflectances.filter(
            ee.Filter.notNull(["closest_image", "B2"])
        )
        # export to Asset: featurecollection
        output_filename = (
            f"open-earth/validation/lai/{foldername}_{site}_reflectances_with_angles"
        )
        task = ee.batch.Export.table.toCloudStorage(
            collection=fc_reflectances_nonull,
            description=f"{foldername}_{site}_s2_reflectances_with_angles",
            bucket="felixspecker",
            fileNamePrefix=output_filename,
        )
        task.start()

        # create placeholder file to remember downloading the export later
        # replace 'merged' with 'export' in the filename
        placeholder_filename = output.replace("merged", "export")
        with open(
            os.path.join(f"{placeholder_filename}.txt"),
            "w",
        ) as f:
            f.write(
                f"Exported at time: {datetime.datetime.now()} \n See results at GCS: {output_filename}"
            )

        logger.info(
            f"Exported GBOV validation data with s2 reflectances to {output_filename}"
        )

    return None


if __name__ == "__main__":
    lai_points()
