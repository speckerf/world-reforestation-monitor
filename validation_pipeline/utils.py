import math
import os
import re
import tempfile
from zipfile import ZipFile

import ee
import geopandas as gpd
import pandas as pd
from loguru import logger


def load_ecoregion_shapefile() -> gpd.GeoDataFrame:
    logger.info("Loading ecoregion shapefile")
    ecoregion_file_path = os.path.join("data", "geometries", "Ecoregions2017.zip")
    shapefile_name = "Ecoregions2017.shp"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the contents of the zip file into the temporary directory
        with ZipFile(ecoregion_file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

            # Construct the path to the shapefile
        shapefile_path = os.path.join(tmpdir, shapefile_name)

        # Read the shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)

    return gdf


def add_angles_from_metadata_to_properties(image: ee.Image) -> ee.Image:
    """
    Enhances the given satellite image with additional bands derived from its metadata angles.

    This function extracts the solar and view angles from the metadata of the satellite image.
    The solar azimuth and zenith angles are directly obtained from the metadata, while the view
    azimuth and zenith angles are calculated as the mean of angles corresponding to various bands.
    These angles are then transformed to cosine values, and the relative azimuth angle is calculated
    as the absolute difference between the view and solar azimuth angles.

    The function also scales the image values by dividing them by 10000, converting them to reflectance values.
    Finally, the transformed angles are added as new bands to the image.

    Parameters:
    - image (ee.Image): The input satellite image which contains metadata from which the angles are extracted.

    Returns:
    - ee.Image: The enhanced image with added bands representing the transformed solar zenith, view zenith,
               and relative azimuth angles.

    Bands Added:
    - 'solar_zenith': Cosine of the solar zenith angle.
    - 'view_zenith': Cosine of the view zenith angle.
    - 'relative_azimuth': Cosine of the absolute difference between the view and solar azimuth angles.

    Example Usage:
    enhanced_image = add_angles_from_metadata_to_bands(original_image)
    """

    # Define the bands for which view angles are extracted from metadata.
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # Extract the solar azimuth and zenith angles from metadata.
    solar_azimuth = image.getNumber("MEAN_SOLAR_AZIMUTH_ANGLE")
    solar_zenith = image.getNumber("MEAN_SOLAR_ZENITH_ANGLE")

    # Calculate the mean view azimuth angle for the specified bands.
    view_azimuth = (
        ee.Array(
            [image.getNumber("MEAN_INCIDENCE_AZIMUTH_ANGLE_%s" % b) for b in bands]
        )
        .reduce(ee.Reducer.mean(), [0])
        .get([0])
    )

    # Calculate the mean view zenith angle for the specified bands.
    view_zenith = (
        ee.Array([image.getNumber("MEAN_INCIDENCE_ZENITH_ANGLE_%s" % b) for b in bands])
        .reduce(ee.Reducer.mean(), [0])
        .get([0])
    )

    # Transform angles to cosine values and calculate relative azimuth angle.
    transformed_angles = {
        "solar_zenith_cos": solar_zenith.multiply(ee.Number(math.pi / 180)).cos(),
        "view_zenith_cos": view_zenith.multiply(ee.Number(math.pi / 180)).cos(),
        "relative_azimuth_cos": view_azimuth.subtract(solar_azimuth)
        .abs()
        .multiply(ee.Number(math.pi / 180))
        .cos(),
        "solar_zenith_rad": solar_zenith.multiply(ee.Number(math.pi / 180)),
        "view_zenith_rad": view_zenith.multiply(ee.Number(math.pi / 180)),
        "relative_azimuth_rad": view_azimuth.subtract(solar_azimuth)
        .abs()
        .multiply(ee.Number(math.pi / 180)),
        "solar_azimuth_rad": solar_azimuth.multiply(ee.Number(math.pi / 180)),
        "view_azimuth_rad": view_azimuth.multiply(ee.Number(math.pi / 180)),
    }

    # # Convert image values to reflectance values.
    # image = image.divide(10000).toFloat() # SUPERSEDED
    logger.trace(
        "Image values are not converted to reflectance values inside add_angles_from_metadata_to_bands, make sure to do this elsewhere."
    )

    # Add the transformed angles as new bands to the image.
    image = image.set("sza", solar_zenith)
    image = image.set("vza", view_zenith)
    image = image.set("phi", view_azimuth.subtract(solar_azimuth).abs())

    # add radians
    # image = image.set('sza_rad', transformed_angles['solar_zenith_rad'])
    # image = image.set('vza_rad', transformed_angles['view_zenith_rad'])
    # image = image.set('phi_rad', transformed_angles['relative_azimuth_rad'])
    # image = image.set('saa_rad', transformed_angles['solar_azimuth_rad'])
    # image = image.set('vaa_rad', transformed_angles['view_azimuth_rad'])

    return image


def add_closest_cloudfree_s2_image_reflectances(feature: ee.Feature) -> ee.Feature:
    max_days_apart = 10

    # Get the point geometry
    point = feature.geometry()

    # Get the date of the feature
    date_insitu = ee.Date(feature.getNumber("system:time_start"))

    # Define the date range for the search
    start_date = date_insitu.advance(-max_days_apart, "day")
    end_date = date_insitu.advance(max_days_apart, "day")

    def date_diff(image):
        img_date = ee.Date(image.get("system:time_start"))
        return image.set(
            "date_difference", img_date.difference(date_insitu, "day").abs()
        )

    csPlus = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date, end_date)
    )
    qa_band = "cs_cdf"
    clear_threshold = 0.6

    def add_point_cloudfree_score(image):
        cloud_score = (
            image.select(qa_band)
            .reduceRegion(
                reducer=ee.Reducer.mean(),  # We expect only one value since it's a point
                geometry=point.buffer(
                    10
                ).bounds(),  # construcs a 20x20m bounding box around the point
                scale=10,  # Adjust as necessary for the resolution of cloud scores
            )
            .get(qa_band)
        )
        # Return a condition that checks if cloud score is less than the threshold
        return image.set("cloud_score", cloud_score)

    # Define the Sentinel-2 collection
    s2_closest = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 50)
        .linkCollection(csPlus, [qa_band])
        .map(lambda img: add_point_cloudfree_score(img))
        .filterMetadata("cloud_score", "greater_than", clear_threshold)
        .map(lambda img: date_diff(img))
        .sort("date_difference")
    )  # .first()

    def add_reflectances_to_metadata(feat: ee.Feature, img: ee.Image) -> ee.Feature:
        band_values = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        s2_closest_reflectances = img.select(band_values).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point.buffer(10).bounds(), scale=10
        )
        feat = feat.set("closest_image", img.get("system:index"))
        feat = feat.set("date_difference", img.get("date_difference"))
        feat = feat.set("cloud_score", img.get("cloud_score"))

        # add angles of observation
        img = add_angles_from_metadata_to_properties(img)
        feat = feat.set("sza", img.get("sza"))
        feat = feat.set("vza", img.get("vza"))
        feat = feat.set("phi", img.get("phi"))

        for band in band_values:
            feat = feat.set(band, s2_closest_reflectances.getNumber(band))
        return feat

    output_feature = ee.Algorithms.If(
        ee.Algorithms.IsEqual(s2_closest.size(), 0),
        ee.Feature(None),
        add_reflectances_to_metadata(feature, s2_closest.first()),
    )
    return output_feature


def is_abbrev(abbrev, text):
    # function that checks if a string could be an abbreviation of another string
    # returns true or false
    pattern = ".*".join(abbrev.lower())
    return re.match("^" + pattern, text.lower()) is not None


# dictionary with site names and abbreviations:
def parse_gbov_site_names(folder_path):
    # list all file names ending txt, split by "_", get thrid element, get unique values
    full_names = set(
        [f.split("_")[2] for f in os.listdir(folder_path) if f.endswith(".txt")]
    )
    # remove the string 'L08' from this list
    full_names = [f for f in full_names if f not in ["L08", "S2B"]]
    # add space character between words in full names which are of the following format: 'SmithsonianConservationBiologyInstitute' (upperCase letter at beginning of word)
    full_names_spaced = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", f) for f in full_names]

    # get abbreviation as fourth element of filename, filter for strings of length 4, get unique values
    abbreviations = set(
        [f.split("_")[3] for f in os.listdir(folder_path) if f.endswith("_README.TXT")]
    )
    abbreviations = [f for f in abbreviations if len(f) == 4]

    ## assert that length of full_names and abbreviations is equal
    assert len(full_names) == len(
        abbreviations
    ), "Number of full names and abbreviations is not equal."

    site_dict = {
        "BART": "BartlettExperimentalForest",
        "BLAN": "BlandyExperimentalFarm",
        "CPER": "CentralPlainsExperimentalRange",
        "DELA": "DeadLake",
        "DSNY": "DisneyWildernessPreserve",
        "GUAN": "GuanicaForest",
        "HAIN": "Hainich",
        "HARV": "HarvardForest",
        "JERC": "JonesEcologicalResearchCenter",
        "JORN": "Jornada",
        "KONA": "KonzaPrairieBiologicalStation",
        "LAJA": "LajasExperimentalStation",
        "LITC": "LitchfieldSavanna",
        "MOAB": "Moab",
        "NIWO": "NiwotRidgeMountainResearchStation",
        "ONAQ": "OnaquiAult",
        "ORNL": "OakRidge",
        "OSBS": "OrdwaySwisherBiologicalStation",
        "SCBI": "SmithsonianConservationBiologyInstitute",
        "SERC": "SmithsonianEnvironmentalResearchCenter",
        "SRER": "SantaRita",
        "STEI": "SteigerwaldtLandServices",
        "STER": "NorthSterling",
        "TALL": "TalladegaNationalForest",
        "TUMB": "Tumbarumba",
        "UNDE": "Underc",
        "VALE": "ValenciaAnchorStation",
        "WOMB": "WombatStringbarkEucalypt",
        "WOOD": "Woodworth",
    }

    # check if all names and abbreviations are in the dictionary
    assert sorted(site_dict.keys()) == sorted(
        abbreviations
    ), "Not all abbreviations are in the dictionary."
    assert sorted(site_dict.values()) == sorted(
        full_names
    ), "Not all full names are in the dictionary."

    return site_dict


class EEExportError(Exception):
    """Exception raised for errors GEE asset exports."""


def wait_for_export(task) -> None:
    """Wait for the export task to finish.

    This method waits for the export task to finish and checks if it has completed or failed.

    Args:
        task (Task): The export task.

    Raises:
        EEExportError: Raised when the export of the asset fails.
    """

    # task = asset_dict["task"]
    prev_state = ""
    task_info = task.status()
    logger.debug(f"Waiting for {task_info['description']} to finish")
    while task.status()["state"] in {
        "UNSUBMITTED",
        "READY",
        "RUNNING",
        "COMPLETED",
    }:
        current_state = task.status()["state"]
        if prev_state != current_state:
            logger.info(current_state)
            prev_state = current_state

        if current_state in {"COMPLETED", "FAILED"}:
            # print(current_state)
            break
    current_state = task.status()["state"]
    if current_state == "FAILED":
        logger.error(current_state)
        raise EEExportError(
            (
                # f"FAILED TO EXPORT ASSET {asset_dict['name']}. \n"
                f"\nerror_message: {task.status()['error_message']}\n"
                f"\nCheck GEE Task Manager for more details: \n"
                f"https://code.earthengine.google.com/tasks"
            )
        )

    logger.debug(f'{task_info["description"]} done')
