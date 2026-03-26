import ee
from loguru import logger

# service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
# credentials = ee.ServiceAccountCredentials(
#     service_account, "auth/gem-eth-analysis-24fe4261f029.json"
# )
# ee.Initialize(credentials, project="ee-speckerfelix")
ee.Initialize(project="ee-speckerfelix")
#


def load_imgcollection(
    trait: str, year: int, export_version: str, resolution: int
) -> ee.ImageCollection:
    logger.debug(
        f"Loading ImageCollection - trait: {trait}, year: {year}, export_version: {export_version}"
    )
    imgc = ee.ImageCollection(
        f"projects/ee-speckerfelix/assets/open-earth/{trait}_predictions-mlp_{resolution}m_{export_version}"
    ).filterDate(f"{year}-01-01", f"{year}-12-31")
    return imgc


def export_to_gdrive(
    trait: str,
    year: int,
    resolution: int,
    export_version: str,
    model_version: str,
    band: str,
    output_crs: str,
    no_data_value_mean: int = -9999,
    no_data_value_stddev: int = -9999,
    no_data_value_count: int = 255,
) -> None:
    logger.info(
        f"Exporting to GDrive - trait: {trait}, year: {year}, export_version: {export_version}, model_version: {model_version}, resolution: {resolution}, band: {band}"
    )

    assert band in ["mean", "stdDev", "count"], f"Band {band} not supported"
    if band == "mean":
        no_data_val = no_data_value_mean
    elif band == "stdDev":
        no_data_val = no_data_value_stddev
    elif band == "count":
        no_data_val = no_data_value_count
    else:
        raise ValueError(f"Band {band} not supported")

    if resolution == 1000:
        # use 100 meter version in GEE, but export at 1000m resolution (resample in export task)
        resolution_gee = 100
    else:
        resolution_gee = resolution

    imgc = load_imgcollection(
        trait=trait, year=year, export_version=export_version, resolution=resolution_gee
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
    filename_method = f"rtm.mlp.{model_version}"
    filename_vertical = "s"  # 'b': below ground, 'a': above ground, 's': surface
    filename_version_code = f"{export_version}"

    # export to DRIVE
    filename = f"{filename_trait}_{filename_method}_{filename_position_probability}_{filename_resolution}_{filename_vertical}_{filename_day_start}_{filename_day_end}_{filename_extent}_{filename_crs}_{filename_version_code}"
    # gdrive_folder = CONFIG_GEE_PIPELINE["GDRIVE_FOLDERS"]["TEMP_FOLDER"]
    gdrive_shared_folder = "gee-service-account-exports"
    gdrive_folder = f"{gdrive_shared_folder}/{trait}-{position_prob_dict[band]}_predictions-mlp_{resolution}m_{export_version}"
    # foldername = f"{filename_trait}_predictions-mlp_{filename_resolution}_{filename_version_code}"

    export_task = ee.batch.Export.image.toDrive(
        image=output_image,
        description=f"Export {filename}",
        folder=gdrive_folder,
        fileNamePrefix=filename,
        scale=resolution,
        region=ee.Geometry.BBox(-180, -60, 180, 85),
        crs=output_crs,
        maxPixels=1e12,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": False, "noData": no_data_val},
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
    years = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    # years = [2025]
    traits = ["laie", "fapar", "fcover"]
    # traits = ["lai", "fapar", "fcover"]
    # traits = ["laie"]
    resolution = 1000
    export_version = "v03"
    model_version = "v02"
    output_crs = "EPSG:4326"
    # band = "mean"
    # bands = ["stdDev", "count"]
    # bands = ["mean", "stdDev", "count"]
    bands = ["mean", "stdDev"]
    for band in bands:
        for trait in traits:
            for year in years:
                image_specs = {
                    "trait": trait,
                    "year": int(year),
                    "resolution": resolution,
                    "export_version": export_version,
                    "model_version": model_version,
                    "band": band,
                    "output_crs": output_crs,
                }
                export_to_gdrive(**image_specs)
