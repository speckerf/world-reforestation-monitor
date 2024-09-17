import concurrent.futures
import os
from datetime import datetime
from functools import reduce

import ee
import numpy as np
import pandas as pd
from loguru import logger

from config.config import get_config
from gee_pipeline.srcGlobal import export_mgrs_tile
from gee_pipeline.utilsAngles import add_angles_from_metadata_to_bands
from gee_pipeline.utilsCloudfree import apply_cloudScorePlus_mask
from gee_pipeline.utilsPhenology import get_start_end_date_phenology_for_ecoregion
from gee_pipeline.utilsPredict import (
    add_random_ensemble_assignment,
    collapse_to_mean_and_stddev,
    eePipelinePredictMap,
)
from gee_pipeline.utilsTiles import get_s2_indices_filtered
from train_pipeline.finalTraining import load_model_ensemble

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


service_account = "crowther-gee@gem-eth-analysis.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(
    service_account, "auth/gem-eth-analysis-24fe4261f029.json"
)
ee.Initialize(credentials, project="ee-speckerfelix")


def test_export_global():
    mgrs_tile_list = ["45S"]
    for mgrs_tile in mgrs_tile_list:
        export_mgrs_tile(mgrs_tile)


if __name__ == "__main__":
    # export_continent()
    # export_helper()
    test_export_global()
    # test_multi_eco()
    pass
