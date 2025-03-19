import ee
from loguru import logger


def add_angles_from_metadata_to_bands(image: ee.Image) -> ee.Image:
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

    # add tts, tto, psi
    image = image.addBands(ee.Image(solar_zenith).toFloat().rename("tts"))
    image = image.addBands(ee.Image(view_zenith).toFloat().rename("tto"))
    image = image.addBands(
        ee.Image(view_azimuth.subtract(solar_azimuth).abs()).toFloat().rename("psi")
    )

    return image


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

    # # Convert image values to reflectance values.
    # image = image.divide(10000).toFloat() # SUPERSEDED
    logger.trace(
        "Image values are not converted to reflectance values inside add_angles_from_metadata_to_bands, make sure to do this elsewhere."
    )

    # Add the transformed angles as new bands to the image.
    image = image.set("sza", solar_zenith)
    image = image.set("vza", view_zenith)
    image = image.set("phi", view_azimuth.subtract(solar_azimuth).abs())

    return image
