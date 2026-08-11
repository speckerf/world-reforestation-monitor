import ee

from config.config import get_config
from gee_pipeline.src_global import export_mgrs_tile

service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def test_export_global():
    config = get_config("gee_pipeline")
    # not global export: only subset of mgrs tiles: see https://hls.gsfc.nasa.gov/products-description/tiling-system/
    # mgrs_tile_list = ["25W"]
    # 29Q to 35 Q
    # mgrs_tile_list =
    # mgrs_tile_list = ["29Q", "30Q", "31Q", "32Q", "33Q", "34Q", "35Q", "44S", "45S", "46S", "44T", "45T", "46T"]  # fmt: skip
    # mgrs_tile_list = ["32M"]
    # mgrs_tile_list = ["16R"]
    # mgrs_tile_list = ["10S"]

    # mgrs_tile_list = [
    #     "36N",
    #     "10T",
    #     "10S",
    #     "15Q",
    #     "16Q",
    #     "34N",
    #     "55G",
    #     "31T",
    #     "24L",
    #     # "29Q",
    #     # "30Q",
    #     # "31Q",
    #     # "18F",
    #     # "31T",
    #     # "44S",
    #     # "45S",
    #     # "46S",
    #     # "44T",
    #     # "45T",
    #     # "46T",
    # ]  # for figures
    # mgrs_tile_list = ["24L", "36N", "50L"]
    # mgrs_tile_list = ["36N", "50L"]
    # mgrs_tile_list = ["18U", "19U"]
    # mgrs_tile_list = ["18F"]
    # mgrs_tile_list = ["29Q", "30Q", "31Q"]  # for figures
    # mgrs_tile_list = ["31T"]  # catalnuy
    # mgrs_tile_list = mgrs_tiles_for_figures
    # mgrs_tile_list = mgrs_tiles_for_figures
    # mgrs_tile_list = ["44S", "45S", "46S", "44T", "45T", "46T"]

    # all laie
    # mgrs_tile_list = ["15Q", "55G", "36N", "34N", "10U", "35L", "32U"]

    # all fapar
    # mgrs_tile_list = ["10T"]

    # all fcover
    mgrs_tile_list = ["18U", "19U"]

    # figure 4B: LAIe all years
    # mgrs_tile_list = ["15Q"]

    # figure 4A: LAIe
    # mgrs_tile_list = ["55G"]

    # figure 4C: fcover
    # mgrs_tile_list = ["18U", "19U"]

    # figure 2A: LAIe 2020 (all uncertainty bands)
    # mgrs_tile_list = ["36N"]

    # figure 2B: FAPAR 2024 10T, all uncertainty bands
    # mgrs_tile_list = ["10T"]

    # figure 2C: FCOVER 2022, 34N, all uncertainty bands
    # mgrs_tile_list = ["34N"]

    # figure S3: Laie 2022, [34N, 10U, 35L, 32U], all uncertainty bands
    # mgrs_tile_list = ["34N", "10U", "35L", "32U"]

    # mgrs_tile_list = ["15Q", "55G", "18U", "19U"]

    # mgrs_tile_list = ["15Q", "55G", "18U", "19U"]

    # these are regions where clamping/masking issue cause recalibration to be unrealistic
    # mgrs_tile_list = ["46T", "43T", "43U", "44U", "39U", "43V", "19G", "19F", "12T"]  # fmt: skip

    for mgrs_tile in mgrs_tile_list:
        export_mgrs_tile(mgrs_tile, config=config)


if __name__ == "__main__":
    test_export_global()
