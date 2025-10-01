import datetime
import os
import sys
from fileinput import filename
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import pandas as pd
from loguru import logger

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

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


# Function to split value and uncertainty if in "+/-" format
def _split_value_uncertainty(x):
    if isinstance(x, str) and "+/-" in x:
        val, unc = x.split("+/-")
        return float(val), float(unc)
    try:
        return float(x), None
    except Exception:
        return x, None  # <- keep original if not numeric


def parse_grounded_eo_fiducial_reference_measurements(filename: str) -> pd.DataFrame:
    cols = [
        "Network",
        "Site",
        "Latitude",
        "Longitude",
        "NLCD",
        "Plot",
        "Date",
        "Method",
        "Processor",
        "Combined_flag",
        "LAI",
        "LAIe",
        "FAPAR",
        "FCOVER",
    ]

    df = pd.read_csv(
        filename,
        usecols=cols,
        na_values=["", "NA", -999],
    )

    # Convert Date column (looks like day/month/year in your preview)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    trait_cols = ["LAI", "LAIe", "FAPAR", "FCOVER"]
    # Create new dict for extracted values

    for col in trait_cols:
        split_results = df[col].apply(_split_value_uncertainty)
        df[col] = split_results.map(lambda x: x[0])  # numeric value
        df[col + "_u"] = split_results.map(lambda x: x[1])  # uncertainty

    return df


def main():
    logger.info(
        "Fetching LAIe/FAPAR/FCOVER from GROUNDED-EO fiducial reference measurements and performing spatio-temporal overlay with Sentinel-2."
    )

    input_filename = "data/validation_pipeline/input/traits_GROUNDED-EO/all_fiducial_reference_measurements.csv"
    output_filename = "merged_lai_GROUNDED-EO.csv"

    logger.info(
        f"Merging GBOV validation data from folder {input_filename} and saving to {output_filename}"
    )

    df = parse_grounded_eo_fiducial_reference_measurements(input_filename)

    # convert to geopandas dataframe with point geometry
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

    # add column ECO_ID from ecoregions gdf
    ecoregions = load_ecoregion_shapefile()
    gdf = gpd.sjoin(
        gdf, ecoregions[["ECO_ID", "geometry"]], how="left", predicate="within"
    ).drop("index_right", axis=1)

    # cast ECO_ID to integer
    gdf["ECO_ID"] = gdf["ECO_ID"].astype("Int32")

    # set crs
    gdf.set_crs(epsg=4326, inplace=True)

    # only keep observation with Combined_flag == 0
    gdf = gdf[gdf["Combined_flag"] == 0]

    # remove observation with NA Latitude or Longitude
    gdf = gdf[~gdf["Latitude"].isna() & ~gdf["Longitude"].isna()]

    # loop over all sites and export reflectances
    for site in gdf["Site"].unique():

        gdf_site = gdf[gdf["Site"] == site]

        # extract the Network name (should be unique per site)
        network = gdf_site["Network"].unique()[0]

        # Add system:time_start in milliseconds since epoch
        gdf_site["system:time_start"] = gdf_site["Date"].astype("int64") // 10**6

        # save to csv for debugging
        # gdf_site.to_csv(f"debug_lai_{site}.csv", index=False)

        fc_site = geemap.geopandas_to_ee(gdf_site)

        fc_reflectances = fc_site.map(
            lambda f: add_closest_cloudfree_s2_image_reflectances(f, network)
        )
        fc_reflectances_nonull = fc_reflectances.filter(
            ee.Filter.notNull(["closest_image", "B2"])
        )
        # export to Asset: featurecollection
        output_filename = f"open-earth/validation/traits_GROUNDED-EO/{site}_s2_matchup"
        task = ee.batch.Export.table.toCloudStorage(
            collection=fc_reflectances_nonull,
            description=f"{site}_s2_matchup",
            bucket="felixspecker",
            fileNamePrefix=output_filename,
        )
        task.start()
        logger.info(f"Exporting {site} to {output_filename}")
    return None


if __name__ == "__main__":
    main()
