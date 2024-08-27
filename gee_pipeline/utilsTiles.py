import random
import string

import ee
import geemap
import geopandas as gpd
from loguru import logger

from config.config import get_config
from gee_pipeline.utils import wait_for_task
from gee_pipeline.utilsPhenology import add_linear_weight

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


def add_group(image):
    orbit = image.getNumber("SENSING_ORBIT_NUMBER")
    tile = image.getString("MGRS_TILE")
    group = ee.String(orbit.format()).cat("_").cat(tile)
    return image.set("group", group)


def groupby_mgrs_orbit_pandas(
    imgc: ee.ImageCollection,
    center_pheno: bool = False,
    start_pheno: ee.Date = None,
    end_pheno: ee.Date = None,
) -> ee.List:

    logger.debug(f"imgc size before grouping and filtering: {imgc.size().getInfo()}")
    imgc = imgc.map(add_group)
    if center_pheno:
        imgc = imgc.map(
            lambda img: add_linear_weight(
                img,
                start_date=start_pheno,
                end_date=end_pheno,
                total_days=end_pheno.difference(start_pheno, "day"),
            )
        )

    # convert imagecollection to pandas data frame: with system:index, group, CLOUDY_PIXEL_PERCENTAGE, and pheno_weoght

    def extract_properties(img):
        return ee.Feature(
            None,
            {
                "s2_index": img.get("system:index"),
                "group": img.get("group"),
                "cloud_pheno_image_weight": img.get("cloud_pheno_image_weight"),
            },
        )

    fc = imgc.map(extract_properties)
    fc_computed = ee.data.computeFeatures({'expression': fc})

    total_features_retrieved = len(fc_computed['features'])
    if 'nextPageToken' in fc_computed:
        logger.debug(f"Partial retrieval of {total_features_retrieved} features. Fetching the rest.")
        fc_list = [fc_computed]
        while 'nextPageToken' in fc_computed:
            fc_computed = ee.data.computeFeatures({'expression': fc, 'pageToken': fc_computed['nextPageToken']})
            fc_list.append(fc_computed)
            total_features_retrieved += len(fc_computed['features'])
            logger.debug(f"Partial retrieval of {total_features_retrieved} features. Fetching the rest.")
        logger.debug(f"Total features retrieved: {total_features_retrieved}")
        fc_concat = {'type': 'FeatureCollection', 'features': [feature for fc in fc_list for feature in fc['features']]}
    else:
        fc_concat = fc_computed

    
    # parse dict to pandas dataframe



    # convert feature collection to pandas dataframe # drop geometry column
    df = gpd.GeoDataFrame.from_features(fc_concat).drop(columns="geometry")

    # filter by group and sort by cloud_pheno_image_weight, limit to 10 images per group
    df_sorted = df.sort_values(by=["group", "cloud_pheno_image_weight"], ascending=[True, True])
    df_grouped = df_sorted.groupby("group").head(10)

    logger.debug(f"imgc size after grouping and filtering: {df_grouped.shape[0]}")

    # return list of system:index
    return df_grouped["s2_index"].tolist()


def groupby_mgrs_orbit_filter_and_export(
    imgc: ee.ImageCollection,
    center_pheno: bool = False,
    start_pheno: ee.Date = None,
    end_pheno: ee.Date = None,
) -> ee.List:
    # Group by orbit and tile using the 'SENSING_ORBIT_NUMBER' and 'MGRS_TILE' properties

    imgc = imgc.map(add_group)
    if center_pheno:
        imgc = imgc.map(
            lambda img: add_linear_weight(
                img,
                start_date=start_pheno,
                end_date=end_pheno,
                total_days=end_pheno.difference(start_pheno, "day"),
            )
        )

    # Aggregate the groups and filter images within each group
    all_groups = imgc.aggregate_array("group").distinct()

    def filter_and_select_indices(group):
        filtered_group = imgc.filter(ee.Filter.eq("group", group)).sort(
            "CLOUDY_PIXEL_PERCENTAGE"
        )

        limited_group = filtered_group.limit(10)
        indices = limited_group.aggregate_array("system:index")
        return indices

    def filter_and_select_indices_phenoweighting(group):
        filtered_group = imgc.filter(ee.Filter.eq("group", group)).sort(
            "cloud_pheno_image_weight"
        )

        limited_group = filtered_group.limit(10)
        indices = limited_group.aggregate_array("system:index")
        return indices

    if center_pheno:
        filtered_collection = all_groups.map(
            filter_and_select_indices_phenoweighting
        ).flatten()
    else:
        filtered_collection = all_groups.map(filter_and_select_indices).flatten()

    # Create a FeatureCollection with empty geometries
    def create_feature(index):
        return ee.Feature(ee.Geometry.Point([0, 0]), {"s2_index": index})

    feature_collection = ee.FeatureCollection(filtered_collection.map(create_feature))

    # export feature collection with random character name
    random_name = "".join(random.choices(string.ascii_lowercase, k=20))

    export_task = ee.batch.Export.table.toAsset(
        collection=feature_collection,
        description=f"s2 imgc grouped filtering: {random_name}",
        assetId=f"{CONFIG_GEE_PIPELINE["GEE_FOLDERS"]["TEMP_FOLDER"]}/{random_name}",
    )
    export_task.start()

    wait_for_task(export_task)

    fc_loaded = ee.FeatureCollection(
        export_task.config["assetExportOptions"]["earthEngineDestination"]["name"]
    )

    return fc_loaded.aggregate_array("s2_index")


if __name__ == "__main__":

    ee.Initialize(project = 'ee-speckerfelix')

    small_geometry = ee.Geometry.Polygon(
        [
            [
                [8.035098798771045, 46.12625275000346],
                [8.035098798771045, 45.59835924137844],
                [9.221622236271045, 45.59835924137844],
                [9.221622236271045, 46.12625275000346],
            ]
        ],
        None,
        False,
    )

    middle_geometry = ee.Geometry.Polygon(
        [
            [
                [7.035098798771045, 47.12625275000346],
                [7.035098798771045, 44.59835924137844],
                [10.221622236271045, 44.59835924137844],
                [10.221622236271045, 47.12625275000346],
            ]
        ],
        None,
        False,
    )

    large_geometry = ee.Geometry.Polygon(
        [
            [
                [6.035098798771045, 53.12625275000346],
                [6.035098798771045, 42.59835924137844],
                [13.221622236271045, 42.59835924137844],
                [13.221622236271045, 53.12625275000346],
            ]
        ],
        None,
        False,
    )

    extra_large_geometry = ee.Geometry.Polygon(
        [
            [
                [1.035098798771045, 60.12625275000346],
                [1.035098798771045, 35.59835924137844],
                [25.221622236271045, 35.59835924137844],
                [25.221622236271045, 60.12625275000346],
            ]
        ],
        None,
        False
    )

    # geometry = middle_geometry
    geometry = extra_large_geometry

    # Define the original ImageCollection
    imgc = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate("2022-04-16", "2022-09-16")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 75))
    )

    logger.info(f"ImageCollection size: {imgc.size().getInfo()}")

    s2_indices_filtered = groupby_mgrs_orbit_pandas(
        imgc,
        center_pheno=True,
        start_pheno=ee.Date("2022-04-16"),
        end_pheno=ee.Date("2022-09-16"),
    )

    del imgc
    imgc = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filter(ee.Filter.inList('system:index', s2_indices_filtered))

    logger.info(f"Filtered ImageCollection size: {imgc.size().getInfo()}")



    # to debug here:
    if False:
        imgc.filter(ee.Filter.eq('MGRS_TILE', '32TLQ')).aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()

        a = imgc.filter(ee.Filter.eq('MGRS_TILE', '32TLQ'))
        cloud_percentages = a.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        days_of_image = a.map(lambda img: ee.Feature(ee.Geometry.Point([0,0]), {'date': img.date().format('YYYY-MM-dd')})).aggregate_array('date').getInfo()
        import pandas as pd
        df = pd.DataFrame({'cloud_percentage': cloud_percentages, 'date': days_of_image})    

        
        from datetime import datetime
        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        df['start_date'] = datetime.strptime('2022-04-16', '%Y-%m-%d')
        df['end_date'] = datetime.strptime('2022-09-16', '%Y-%m-%d')

        df['days_from_start'] = (df['date'] - df['start_date']).dt.days
        df['days_to_end'] = (df['date'] - df['end_date']).dt.days.abs()
        df['min_days_to_start_or_end'] = df[['days_from_start', 'days_to_end']].min(axis=1)
        df['total_days'] = (df['end_date'] - df['start_date']).dt.days
        df['pheno_weight'] = df['min_days_to_start_or_end'] / (df['total_days'] / 2)
        df['pheno_weight_inverted'] = (1 - df['pheno_weight']) / 2
        df['cloud_weight'] = df['cloud_percentage'] / 100
        df['cloud_pheno_weight_combined'] = df['cloud_weight'] + df['pheno_weight_inverted']


        # pheno_cloud_weight = cloud_percentage/100 + (1 - weight)/2
        # weight is 1 at the midpoint, 0 at the start and end


