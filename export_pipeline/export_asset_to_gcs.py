import ee
from loguru import logger

from config.config import get_config

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")

ee.Initialize()


def load_imgcollection(
    trait: str, year: int, version: str, resolution: int
) -> ee.ImageCollection:
    logger.debug(
        f"Loading ImageCollection - trait: {trait}, year: {year}, version: {version}"
    )
    imgc = ee.ImageCollection(
        f"projects/ee-speckerfelix/assets/open-earth/{trait}_predictions-mlp_{resolution}m_{version}"
    ).filterDate(f"{year}-01-01", f"{year}-12-31")
    return imgc


def export_to_gcs(
    trait: str, year: int, resolution: int, version: str, band: str
) -> None:
    logger.info(
        f"Exporting to GCS - trait: {trait}, year: {year}, version: {version}, resolution: {resolution}, band: {band}"
    )
    output_crs = CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["EPSG"]
    no_data_val = CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["NO_DATA_VALUE"]

    imgc = load_imgcollection(
        trait=trait, year=year, version=version, resolution=resolution
    ).select(f"{trait}_{band}")

    output_image = imgc.mosaic()

    position_prob_dict = {
        "mean": "mean",
        "stdDev": "std",
        "count": "count",
    }

    # create filename according to filename convention:
    # convert 'EPSG:3035' to 'epsg.3035'
    filename_trait = trait.lower()  # e.g. EWT
    filename_position_probability = position_prob_dict[band]  # e.g. mean
    filename_crs = output_crs.replace("EPSG:", "epsg.")
    filename_extent = "go"
    filename_resolution = f"{resolution}m"
    filename_day_start = f"{year}0101"
    filename_day_end = f"{year}1231"
    filename_method = f"rtm.mlp"
    filename_vertical = "s"  # 'b': below ground, 'a': above ground, 's': surface
    filename_version_code = f"{version}"

    # export
    filename = f"{filename_trait}_{filename_method}_{filename_position_probability}_{filename_resolution}_{filename_vertical}_{filename_day_start}_{filename_day_end}_{filename_extent}_{filename_crs}_{filename_version_code}"
    export_task = ee.batch.Export.image.toCloudStorage(
        image=output_image,
        description=f"Export {filename}",
        region=ee.Geometry.BBox(-180, -88, 180, 99),
        bucket=CONFIG_GEE_PIPELINE["GCLOUD_FOLDERS"]["BUCKET"],
        scale=resolution,
        crs=output_crs,
        fileNamePrefix=f"{CONFIG_GEE_PIPELINE['GCLOUD_FOLDERS']['EXPORT_FOLDER_INTERMEDIATE']}/{filename}/",
        maxPixels=1e12,
        formatOptions={"cloudOptimized": False, "noData": no_data_val},
    )
    export_task.start()

    logger.info(f"Export task started: {export_task.id}")


if __name__ == "__main__":
    image_specs = {
        "trait": CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["TRAIT"],
        "year": int(CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["YEAR"]),
        "resolution": int(CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["RESOLUTION"]),
        "version": CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["VERSION"],
        "band": CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["BAND"],
    }

    export_to_gcs(**image_specs)
