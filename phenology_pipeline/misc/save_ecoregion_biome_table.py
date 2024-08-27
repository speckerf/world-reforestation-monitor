import os
import tempfile
from zipfile import ZipFile

import geopandas as gpd
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

        # Read the shapefile using geopandas / ignore geometries and only load ECO_ID and BIOME_NUM columns
        df = gpd.read_file(shapefile_path, ignore_geometry=True)

    return df


def main():
    df = load_ecoregion_shapefile()
    save_path = os.path.join("data", "misc", "ecoregion_biome_table.csv")
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
