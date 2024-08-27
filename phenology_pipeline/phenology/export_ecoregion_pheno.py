import ee
from loguru import logger

ee.Initialize(project="ee-speckerfelix")


def main():
    # Load the ecoregions feature collection
    ecoRegions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")

    # Load the phenology image
    avg_image = ee.Image(
        "projects/ee-speckerfelix/assets/phenology/global_pheno_cycle1_min5ona_avg_2001_2021_no_artificial_with_amplitude"
    )

    # Get the CRS and transformation from the image
    pheno_crs = avg_image.projection().getInfo()["crs"]
    pheno_transform = avg_image.projection().getInfo()["transform"]

    # Define the bands to process
    bands_pheno = [
        "Greenup_1",
        "MidGreenup_1",
        "Maturity_1",
        "Peak_1",
        "Senescence_1",
        "MidGreendown_1",
        "Dormancy_1",
    ]
    bands_other = ["EVI_Minimum_1", "EVI_Amplitude_1", "EVI_Area_1"]

    # Exclude ecoregions by ECO_ID (example with exclusion list containing [0])
    exclusion_list = [0]
    ecoRegions = ecoRegions.filter(ee.Filter.inList("ECO_ID", exclusion_list).Not())

    def process_feature(feature):
        # Calculate the spatial mode and mean for phenology bands
        spatial_mode = avg_image.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=feature.geometry(),
            crs=pheno_crs,
            crsTransform=pheno_transform,
            maxPixels=1e6,
            bestEffort=True,
        )

        spatial_mean = avg_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            crs=pheno_crs,
            crsTransform=pheno_transform,
            maxPixels=1e6,
            bestEffort=True,
        )

        # Update feature with the spatial mode for phenology bands
        for bandName in bands_pheno:
            value = spatial_mode.get(bandName)
            feature = feature.set(bandName, value)

        for bandName in bands_other:
            value = spatial_mean.get(bandName)
            feature = feature.set(bandName, value)

        return feature.setGeometry(None)

    # Map the function over the ecoregions
    global_by_ecoregion = ecoRegions.map(process_feature)

    all_eco_ids = global_by_ecoregion.aggregate_array("ECO_ID").getInfo()

    for eco_id in all_eco_ids:
        logger.info(f"Exporting ecoregion {eco_id}")
        # Example of how to filter and export one ecoregion
        selected_ecoregion = global_by_ecoregion.filter(ee.Filter.eq("ECO_ID", eco_id))

        file_name_prefix = f"open-earth/phenology/artificial_masked_w_amplitude_singleeco/pheno_eco_{eco_id}"
        task = ee.batch.Export.table.toCloudStorage(
            collection=selected_ecoregion,
            description=f"pheno_eco_{eco_id}",
            bucket="felixspecker",
            fileNamePrefix=file_name_prefix,
            fileFormat="CSV",
        )
        task.start()


if __name__ == "__main__":
    main()
