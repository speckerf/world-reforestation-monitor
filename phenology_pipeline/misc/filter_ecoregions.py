import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from validation_pipeline.utils import load_ecoregion_shapefile


def main():
    ecoregions = load_ecoregion_shapefile()
    ecoregions_subset = ecoregions[
        ["ECO_ID", "ECO_NAME", "BIOME_NUM", "REALM", "SHAPE_AREA", "geometry"]
    ]
    # filter out antarctic ecoregions and ecoregions that are too small
    ecoregions_antarctic = ecoregions_subset[
        ecoregions_subset["REALM"] == "Antarctica"
    ].drop("geometry", axis=1)
    ecoregions_too_small = ecoregions_subset[
        ecoregions_subset["SHAPE_AREA"] < 0.005
    ].drop("geometry", axis=1)

    # add column 'reason' to explain why ecoregion is excluded
    ecoregions_antarctic["reason"] = "Antarctic"
    ecoregions_too_small["reason"] = "Too small: < 500 km^2"

    # also add all ecoregions for which no phenology period was found
    pheno_df = pd.read_csv(
        os.path.join(
            "data",
            "gee_pipeline",
            "inputs",
            "phenology",
            "artificial_masked_w_amplitude_singleeco.csv",
        )
    )
    # find all ECO_IDs with missing start_season and end_season
    missing_pheno = pheno_df[
        pheno_df["start_season"].isna() | pheno_df["end_season"].isna()
    ]

    ecoregions_no_pheno = ecoregions_subset[
        ecoregions_subset["ECO_ID"].isin(missing_pheno["ECO_ID"])
    ].drop("geometry", axis=1)
    ecoregions_no_pheno["reason"] = "No phenology period found"

    # combine all ecoregions to exclude
    ecoregions_to_exclude = pd.concat(
        [ecoregions_antarctic, ecoregions_too_small, ecoregions_no_pheno]
    )
    # remove duplicates (keep first)
    ecoregions_to_exclude = ecoregions_to_exclude.drop_duplicates(subset="ECO_ID")

    ecoregions_to_exclude.to_csv(
        "config/ecoregions_to_exclude_automatic.csv", index=False
    )

    # load manually excluded ecoregions
    ecoregions_manual = pd.read_csv("config/ecoregions_to_exclude_manual.csv")

    # combine automatic and manual exclusions
    ecoregions_to_exclude_all = pd.concat([ecoregions_to_exclude, ecoregions_manual])
    ecoregions_to_exclude_all = ecoregions_to_exclude_all.drop_duplicates(
        subset="ECO_ID"
    )
    ecoregions_to_exclude_all.to_csv(
        "config/ecoregions_to_exclude_all.csv", index=False
    )


if __name__ == "__main__":
    main()
