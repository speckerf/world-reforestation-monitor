import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from validation_pipeline.utils import load_ecoregion_shapefile


def main():
    ecoregions_set_full_year = yaml.load(
        open(os.path.join("config", "ecoregions_pheno_manual_full_year.yaml"), "r"),
        Loader=yaml.FullLoader,
    )["ecoregions"]

    ecoregions = load_ecoregion_shapefile()
    ecoregions_subset = ecoregions[["ECO_ID", "ECO_NAME", "BIOME_NUM"]]

    pheno_df = pd.read_csv(
        os.path.join(
            "data",
            "phenology_pipeline",
            "outputs",
            "artificial_masked_w_amplitude_singleeco.csv",
        )
    )[["ECO_ID", "start_season", "end_season"]]

    ecoregions_manual = pd.DataFrame(ecoregions_set_full_year, columns=["ECO_ID"])
    ecoregions_manual = ecoregions_manual.merge(
        ecoregions_subset, on="ECO_ID", how="left"
    )

    ecoregions_manual = ecoregions_manual.merge(pheno_df, on="ECO_ID", how="left")

    # rename start_season and end_season to old_start_season and old_end_season
    ecoregions_manual = ecoregions_manual.rename(
        columns={"start_season": "old_start_season", "end_season": "old_end_season"}
    )

    # assign 01-01 to start_season and 12-31 to end_season
    ecoregions_manual["start_season"] = "01-01"
    ecoregions_manual["end_season"] = "12-31"

    ecoregions_manual["reason"] = "Manually set to full year due to high cloud cover"

    ecoregions_manual.to_csv(
        os.path.join(
            "data",
            "phenology_pipeline",
            "outputs",
            "ecoregions_pheno_manual_full_year.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
