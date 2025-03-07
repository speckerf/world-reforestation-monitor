import string

import ee
from loguru import logger

# Initialize the Earth Engine library.

service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def get_image_list(collection_name, year: int):
    """Retrieve all image IDs from the specified image collection."""
    collection = ee.ImageCollection(collection_name).filterDate(
        f"{year}-01-01", f"{year}-12-31"
    )
    return collection.aggregate_array("system:index").getInfo()


def group_images_by_epsg(image_list):
    """Group image IDs by their EPSG/UTM code."""
    utm_images = {}
    for image_id in image_list:
        # Extract the UTM zone and EPSG code from the image ID
        epsg_code = image_id.split("_")[-2]  # e.g., 'epsg-32735'

        if epsg_code not in utm_images:
            utm_images[epsg_code] = []

        utm_images[epsg_code].append(image_id)
    return utm_images


def create_system_index(image_ids):
    """Create a system:index for the new merged image."""
    first_image_id = image_ids[0]
    (
        trait,
        model,
        probabilty_mode,
        resolution,
        a,
        startdata,
        enddate,
        utm_tile,
        projection,
        version,
    ) = first_image_id.split("_")

    # Extract the UTM zone (T35) and the EPSG code (32635)
    utm_zone = utm_tile.split("-")[-1][0:3]  # e.g., 'T35M' becomes 'T35'

    # Generate the new system:index
    new_system_index = f"{trait}_{model}_{probabilty_mode}_{resolution}_{a}_{startdata}_{enddate}_{utm_zone}_{projection}_{version}"

    # resolution = 100
    resolution_return = resolution.split("m")[0]

    epsg_code = projection.replace("-", ":").upper()

    return (
        new_system_index,
        resolution_return,
        epsg_code,
    )


def mosaic_and_export(utm_code, image_ids, output_collection):
    """Mosaic images by UTM zone and export them to the specified collection."""
    images = ee.ImageCollection(
        [ee.Image(f"{input_collection}/{img_id}") for img_id in image_ids]
    )
    mosaic = images.mosaic()

    # Generate the new system:index
    new_system_index, resolution, epsg_code = create_system_index(image_ids)

    export_path = f"{output_collection}/{new_system_index}"

    # TODO: test this code
    #  set properties: including year, system:time_start, system:time_end, system:index
    mosaic = mosaic.set("system:index", new_system_index)
    mosaic = mosaic.set(
        "system:time_start", images.first().getNumber("system:time_start")
    )
    mosaic = mosaic.set("system:time_end", images.first().getNumber("system:time_end"))
    mosaic = mosaic.set("year", images.first().get("year"))

    task = ee.batch.Export.image.toAsset(
        image=mosaic,
        description=f"Export {utm_code} mosaic",
        assetId=export_path,
        scale=int(resolution),
        crs=epsg_code,
        region=images.geometry().bounds(),
        maxPixels=1e13,
    )
    task.start()
    logger.debug(
        f"Started export for {utm_code} to {export_path} with num_images: {len(image_ids)}"
    )


def process_image_collection(input_collection, output_collection, year):
    """Main processing function that orchestrates the listing, grouping, and exporting."""
    logger.info(f"Retrieving images from collection: {input_collection}")
    image_list = get_image_list(input_collection, year)

    logger.info(f"Grouping {len(image_list)} images by EPSG code...")
    utm_images = group_images_by_epsg(image_list)

    logger.debug(
        f"Found {len(utm_images)} UTM zones. Ready to export to: {output_collection}"
    )
    confirmation = "yes"  # input("Do you want to start the export? (yes/no): ")

    if confirmation.lower() == "yes":
        for utm_code, image_ids in utm_images.items():
            mosaic_and_export(utm_code, image_ids, output_collection)
        logger.debug("All export tasks have been started.")
    else:
        logger.error("Operation canceled.")


if __name__ == "__main__":
    # Define the input and output image collections
    input_collection = (
        "projects/ee-speckerfelix/assets/open-earth/lai_predictions-mlp_100m_v2"
    )
    output_collection = "projects/ee-speckerfelix/assets/open-earth/lai_predictions-mlp_100m_v2-utm-merged"
    year = 2018

    process_image_collection(input_collection, output_collection, year)
