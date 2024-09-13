import os
import random
import string

import ee
import geemap
import geopandas as gpd
import pandas as pd
from loguru import logger
from pyproj import CRS

from config.config import get_config
from gee_pipeline.utils import wait_for_task
from gee_pipeline.utilsCloudfree import apply_cloudScorePlus_mask
from gee_pipeline.utilsPhenology import add_linear_weight

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


def get_s2_indices_filtered(
    ecoregion_boundary: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    is_full_year: bool = False,
    mgrs_tiles: list = None,
) -> pd.DataFrame:
    # load s2 data
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ecoregion_boundary)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.inList("MGRS_TILE", mgrs_tiles))
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
        start_pheno=start_date,
        end_pheno=end_date,
        is_full_year=is_full_year,
    )
    return s2_indices_filtered


def add_group(image):
    orbit = image.getNumber("SENSING_ORBIT_NUMBER")
    tile = image.getString("MGRS_TILE")
    group = ee.String(orbit.format()).cat("_").cat(tile)
    return image.set("group", group)


def get_epsg_code_from_mgrs(mgrs_zone_number: str):
    """
    Get the EPSG code for a UTM zone from its MGRS zone number.

    Parameters:
    - zone_number (int): The MGRS zone number. Four digit string, e.g. 'T40K'
        - if last digit is from N to Z: Northern hemisphere
        - if last digit is from A to M: Southern hemisphere

    Returns:
    - int: The EPSG code.
    """
    assert len(mgrs_zone_number) == 4, "MGRS zone number must be a four digit string."
    if mgrs_zone_number[-1] in string.ascii_uppercase[13:26]:
        southern_hemisphere = False
    elif mgrs_zone_number[-1] in string.ascii_uppercase[0:13]:
        southern_hemisphere = True
    else:
        raise ValueError("Invalid MGRS zone number.")
    zone_number = int(mgrs_zone_number[1:3])
    crs = CRS.from_dict(
        {
            "proj": "utm",
            "zone": zone_number,
            "south": southern_hemisphere,
            "datum": "WGS84",
            "units": "m",
        }
    )

    return crs.to_epsg()


def add_ndvi_weight(image: ee.Image) -> ee.Image:

    worldcover = ee.ImageCollection("ESA/WorldCover/v100").first()
    natural_classes = [10, 20, 30, 60, 70, 80, 90, 95, 100]
    natural_mask = worldcover.remap(
        natural_classes, ee.List.repeat(1, len(natural_classes)), 0
    )

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndvi = ndvi.updateMask(natural_mask)

    mean_ndvi = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=image.geometry(), scale=1000
    ).getNumber("ndvi")
    mean_ndvi = ee.Algorithms.If(ee.Algorithms.IsEqual(mean_ndvi, None), -1, mean_ndvi)

    cloudy_pixel_percentage = image.getNumber("CLOUDY_PIXEL_PERCENTAGE").divide(100)
    ndvi_weight = ee.Number(1).subtract(mean_ndvi).multiply(2)
    cloud_pheno_weight_combined = cloudy_pixel_percentage.add(ndvi_weight)

    return image.set(
        "cloud_pheno_image_weight",
        cloud_pheno_weight_combined,
        "mean_ndvi",
        mean_ndvi,
        "ndvi_weight",
        ndvi_weight,
        "cloudy_pixel_percentage",
        cloudy_pixel_percentage,
    )


def add_evi_weight(image: ee.Image) -> ee.Image:

    worldcover = ee.ImageCollection("ESA/WorldCover/v100").first()
    natural_classes = [10, 20, 30, 60, 70, 80, 90, 95, 100]
    natural_mask = worldcover.remap(
        natural_classes, ee.List.repeat(1, len(natural_classes)), 0
    )

    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8").divide(10000),
            "RED": image.select("B4").divide(10000),
            "BLUE": image.select("B2").divide(10000),
        },
    ).rename("evi")
    evi = evi.updateMask(natural_mask)

    mean_evi = evi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=image.geometry(), scale=1000
    ).getNumber("evi")
    mean_evi = ee.Algorithms.If(ee.Algorithms.IsEqual(mean_evi, None), -1, mean_evi)

    cloudy_pixel_percentage = image.getNumber("CLOUDY_PIXEL_PERCENTAGE").divide(100)
    evi_weight = ee.Number(1).subtract(mean_evi).multiply(2)
    cloud_pheno_weight_combined = cloudy_pixel_percentage.add(evi_weight)

    return image.set(
        "cloud_pheno_image_weight",
        cloud_pheno_weight_combined,
        "mean_evi",
        mean_evi,
        "evi_weight",
        evi_weight,
        "cloudy_pixel_percentage",
        cloudy_pixel_percentage,
    )


def groupby_mgrs_orbit_pandas(
    imgc: ee.ImageCollection,
    start_pheno: ee.Date = None,
    end_pheno: ee.Date = None,
    is_full_year: bool = False,
) -> ee.List:

    imgc = imgc.map(add_group)

    # mask out clouds and shadows using cloudscore plus
    imgc = apply_cloudScorePlus_mask(imgc)

    # add pheno distance weight / new with evi instead of ndvi
    imgc = imgc.map(add_evi_weight)

    # convert imagecollection to pandas data frame: with system:index, group, CLOUDY_PIXEL_PERCENTAGE, and pheno_weoght

    def extract_properties(img):
        return ee.Feature(
            None,
            {
                "s2_index": img.getString("system:index"),
                "group": img.getString("group"),
                "cloud_pheno_image_weight": img.getNumber("cloud_pheno_image_weight"),
                "cloudy_pixel_percentage": img.getNumber("cloudy_pixel_percentage"),
                "pheno_distance_weight": img.getNumber("pheno_distance_weight"),
                "mean_ndvi": img.getNumber("mean_ndvi"),
                "ndvi_weight": img.getNumber("ndvi_weight"),
                "mean_evi": img.getNumber("mean_evi"),
                "evi_weight": img.getNumber("evi_weight"),
            },
        )

    fc = imgc.map(extract_properties)
    fc_computed = ee.data.computeFeatures({"expression": fc})

    total_features_retrieved = len(fc_computed.get("features", {}))
    if total_features_retrieved == 0:
        logger.error(f"Empty feature collection; after grouping.")
        return []
    if "nextPageToken" in fc_computed:
        logger.debug(
            f"Partial retrieval of {total_features_retrieved} features. Fetching the rest."
        )
        fc_list = [fc_computed]
        while "nextPageToken" in fc_computed:
            fc_computed = ee.data.computeFeatures(
                {"expression": fc, "pageToken": fc_computed["nextPageToken"]}
            )
            fc_list.append(fc_computed)
            total_features_retrieved += len(fc_computed["features"])
            logger.debug(
                f"Partial retrieval of {total_features_retrieved} features. Fetching the rest."
            )
        logger.debug(f"Total features retrieved: {total_features_retrieved}")
        fc_concat = {
            "type": "FeatureCollection",
            "features": [feature for fc in fc_list for feature in fc["features"]],
        }
    else:
        logger.debug(f"Total features retrieved: {total_features_retrieved}")
        fc_concat = fc_computed

    # parse dict to pandas dataframe

    # convert feature collection to pandas dataframe # drop geometry column
    df = gpd.GeoDataFrame.from_features(fc_concat).drop(columns="geometry")

    # filter by group and sort by cloud_pheno_image_weight, limit to 10 images per group
    df_sorted = df.sort_values(
        by=["group", "cloud_pheno_image_weight"], ascending=[True, True]
    )
    df_grouped = df_sorted.groupby("group").head(
        CONFIG_GEE_PIPELINE["CLOUD_FILTERING"]["MAX_IMAGES_PER_GROUP"]
    )

    logger.debug(f"imgc size after grouping and filtering: {df_grouped.shape[0]}")

    # return list of system:index
    return df_grouped["s2_index"].tolist()


def save_mgrs_tiles_for_ecoregion(eco_id):
    # get sentinel-2 mgrs tiles in this ecoregion
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    ecoregion_geometry = ecoregions.filter(ee.Filter.eq("ECO_ID", eco_id)).geometry()

    s2_imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2023-06-01", "2023-06-30")
        .filterBounds(ecoregion_geometry)
    )

    mgrs_tiles = s2_imgc.aggregate_array("MGRS_TILE").getInfo()
    unique_mgrs_tiles = list(set(mgrs_tiles))

    # save to csv
    df = pd.DataFrame(unique_mgrs_tiles, columns=["mgrs_tile"])

    filename = f"mgrs_tiles_ecoregion_{eco_id}.csv"
    foldername = os.path.join("data", "gee_pipeline", "outputs", "mgrs_tiles")

    os.makedirs(foldername, exist_ok=True)

    with open(os.path.join(foldername, filename), "w") as f:
        df.to_csv(f, index=False)


def save_mgrs_tiles_all_ecoregions():
    from concurrent.futures import ThreadPoolExecutor

    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    eco_ids = ecoregions.aggregate_array("ECO_ID").getInfo()

    # Use ThreadPoolExecutor to execute the loop in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(save_mgrs_tiles_for_ecoregion, eco_ids)


if __name__ == "__main__":

    ee.Initialize(project="ee-speckerfelix")

    save_mgrs_tiles_all_ecoregions()
    # get_s2_indices_filtered(
