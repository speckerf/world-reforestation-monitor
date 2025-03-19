import ee

from config.config import get_config
from gee_pipeline.srcGlobal import export_mgrs_tile

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def test_export_global():
    # not global export: only subset of mgrs tiles: see https://hls.gsfc.nasa.gov/products-description/tiling-system/
    # mgrs_tile_list = ["25W"]
    # 29Q to 35 Q
    # mgrs_tile_list = ["29Q", "30Q", "31Q", "32Q", "33Q", "34Q", "35Q"]

    # mgrs_tile_list = ["10T", "10S", "15Q", "16Q", "34N", "55G", "31T", "24L", ]  # for figures
    # mgrs_tile_list = ["24L", "36N", "50L"]
    # mgrs_tile_list = ["36N", "50L"]
    # mgrs_tile_list = ["18U", "19U"]
    mgrs_tile_list = ["18F"]
    # mgrs_tile_list = ["29Q", "30Q", "31Q"]  # for figures
    # mgrs_tile_list = ["31T"]  # catalnuy
    # mgrs_tile_list = mgrs_tiles_for_figures
    # mgrs_tile_list = mgrs_tiles_for_figures
    # mgrs_tile_list = ["44S", "45S", "46S", "44T", "45T", "46T"]
    for mgrs_tile in mgrs_tile_list:
        export_mgrs_tile(mgrs_tile)


if __name__ == "__main__":
    test_export_global()
