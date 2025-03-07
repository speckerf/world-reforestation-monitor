import ee
from loguru import logger

from config.config import get_config

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")

service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")
# ee.Initialize()


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
    if band == "mean":
        no_data_val = CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["NO_DATA_VALUE_MEAN"]
    elif band == "stdDev":
        no_data_val = CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["NO_DATA_VALUE_STDDEV"]
    elif band == "count":
        no_data_val = CONFIG_GEE_PIPELINE["EXPORT_PARAMS"]["NO_DATA_VALUE_COUNT"]
    else:
        raise ValueError(f"Band {band} not supported")

    imgc = load_imgcollection(
        trait=trait, year=year, version=version, resolution=resolution
    ).select(f"{trait}_{band}")

    output_image = imgc.mosaic().unmask(no_data_val)

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

    # export to DRIVE
    filename = f"{filename_trait}_{filename_method}_{filename_position_probability}_{filename_resolution}_{filename_vertical}_{filename_day_start}_{filename_day_end}_{filename_extent}_{filename_crs}_{filename_version_code}"
    # gdrive_folder = CONFIG_GEE_PIPELINE["GDRIVE_FOLDERS"]["TEMP_FOLDER"]
    gdrive_folder = (
        f"{trait}-{position_prob_dict[band]}_predictions-mlp_{resolution}m_{version}"
    )
    # foldername = f"{filename_trait}_predictions-mlp_{filename_resolution}_{filename_version_code}"

    export_task = ee.batch.Export.image.toDrive(
        image=output_image,
        description=f"Export {filename}",
        folder=f"{gdrive_folder}",
        fileNamePrefix=f"{filename}",
        scale=resolution,
        region=ee.Geometry.BBox(-180, -60, 180, 85),
        crs=output_crs,
        maxPixels=1e12,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": no_data_val},
    )
    export_task.start()

    # filename = f"{filename_trait}_{filename_method}_{filename_position_probability}_{filename_resolution}_{filename_vertical}_{filename_day_start}_{filename_day_end}_{filename_extent}_{filename_crs}_{filename_version_code}"
    # bucket = CONFIG_GEE_PIPELINE["GCLOUD_FOLDERS"]["BUCKET"]
    # subfoldername = f"{filename_trait}_predictions-mlp_{filename_resolution}_{filename_version_code}"
    # filename_full_path = f"{CONFIG_GEE_PIPELINE['GCLOUD_FOLDERS']['EXPORT_FOLDER_INTERMEDIATE']}/{subfoldername}/{filename}"

    # export_task = ee.batch.Export.image.toCloudStorage(
    #     image=output_image,
    #     description=f"Export {filename}",
    #     region=ee.Geometry.BBox(-180, -60, 180, 85),
    #     bucket=bucket,
    #     scale=resolution,
    #     crs=output_crs,
    #     fileNamePrefix=filename_full_path,
    #     maxPixels=1e12,
    #     formatOptions={"cloudOptimized": True, "noData": no_data_val},
    # )
    # export_task.start()

    # logger.info(f"Export task started: {export_task.id}")


if __name__ == "__main__":
    years = ["2019", "2020", "2021", "2022", "2023", "2024"]
    # years = [2020]
    traits = ["fapar", "fcover"]
    # traits = ["lai", "fapar", "fcover"]
    # traits = ["lai"]
    resolution = 1000
    version = "v01"
    # band = "mean"
    # bands = ["stdDev", "count"]
    bands = ["mean", "stdDev", "count"]
    for band in bands:
        for trait in traits:
            for year in years:
                image_specs = {
                    "trait": trait,
                    "year": int(year),
                    "resolution": resolution,
                    "version": version,
                    "band": band,
                }
                export_to_gcs(**image_specs)
