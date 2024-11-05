import glob

import ee
import ee.batch
import geopandas as gpd
import pandas as pd
from loguru import logger

# initialize with ee service account
service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")

"""
This script is used to determine for every MGRS tile the relevant Sentinel-2 orbits numbers and save them to a file.
- MGRS tiles are defined by the USGS and are used to divide the world into 100km x 100km tiles.
- Each tile is covered by one or more Sentinel-2 orbits, but can be efficiently represented by usually at most 2 orbits.
    - Get mean NODATA_PIXEL_PERCENTAGE for each orbit, only keep the top two orbits with the lowest mean NODATA_PIXEL_PERCENTAGE.
- This reduces the number of images that need to be processed for each tile.
"""


def get_all_s2_tiles():
    imgc = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(
        "2020-06-01", "2020-06-07"
    )
    return ee.List(imgc.aggregate_array("MGRS_TILE").distinct())


def get_relevant_orbits_for_tile(tile):
    imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2020-06-01", "2020-12-31")
        .filter(ee.Filter.eq("MGRS_TILE", tile))
    )

    orbits = ee.List(imgc.aggregate_array("SENSING_ORBIT_NUMBER").distinct())

    # Calculate mean NODATA_PIXEL_PERCENTAGE for each orbit
    orbit_stats = ee.FeatureCollection(
        orbits.map(
            lambda orbit: ee.Feature(
                None,
                {
                    "orbit": orbit,
                    "mean_nodata": imgc.filter(
                        ee.Filter.eq("SENSING_ORBIT_NUMBER", orbit)
                    ).aggregate_mean("NODATA_PIXEL_PERCENTAGE"),
                },
            )
        )
    )

    orbit_stats_sorted = orbit_stats.sort("mean_nodata", True)

    # all orbits
    all_orbits = orbit_stats_sorted.aggregate_array("orbit")
    orbits_mean_no_data = orbit_stats_sorted.aggregate_array("mean_nodata")

    # Get the orbit numbers
    orbits_to_keep = orbit_stats_sorted.limit(2).aggregate_array("orbit")

    # Return as a feature for the given MGRS tile
    return ee.Feature(
        None,
        {
            "MGRS_TILE": tile,
            "ORBITS_TO_KEEP": orbits_to_keep,
            "ALL_ORBITS": all_orbits,
            "ORBIT_MEAN_NODATA_PERCENTAGE": orbits_mean_no_data,
        },
    )


def export():
    mgrs_tiles = get_all_s2_tiles()
    # Create a FeatureCollection of relevant orbits for each MGRS tile

    # export batches of 1000 tiles
    len_mgrs_tiles = mgrs_tiles.size().getInfo()
    logger.info(f"Total number of MGRS tiles: {len_mgrs_tiles}")
    for i in range(0, len_mgrs_tiles, 1000):
        logger.info(f"Exporting tiles {i} to {min(i+1000, len_mgrs_tiles)}")

        mgrs_tiles_batch = mgrs_tiles.slice(i, i + 1000)
        fc_results = ee.FeatureCollection(
            mgrs_tiles_batch.map(lambda tile: get_relevant_orbits_for_tile(tile))
        )

        logger.debug("Export to CSV Google Cloud Storage")
        ee.batch.Export.table.toCloudStorage(
            bucket="felixspecker",
            collection=fc_results,
            description=f"s2_orbits_per_mgrs_tile_{i}",
            fileNamePrefix=f"open-earth/various/s2_orbits_per_mgrs_tile/s2_orbits_per_mgrs_tile_{i}",
            fileFormat="CSV",
        ).start()


def merge():
    # list files in 'data/gee_pipeline/outputs/s2_orbits_per_mgrs_tile' directory with .csv extension
    files = glob.glob("data/gee_pipeline/outputs/s2_orbits_per_mgrs_tile/*.csv")

    # read all files into a single dataframe
    df = pd.concat([pd.read_csv(file) for file in files])

    # convert column 'ORBITS_TO_KEEP' which is list of float, to list of int
    df["ORBITS_TO_KEEP"] = df["ORBITS_TO_KEEP"].apply(
        lambda x: [int(float(i)) for i in x.strip("[]").split(",")]
    )

    # create larger tile: e.g. 30TUM to 30T

    df["MGRS_TILE_LARGE"] = df["MGRS_TILE"].str[:3]

    # save to a single csv file
    df.to_csv(
        "data/gee_pipeline/outputs/s2_orbits_per_mgrs_tile_merged.csv", index=False
    )


if __name__ == "__main__":
    # export()

    merge()
    # merge()
