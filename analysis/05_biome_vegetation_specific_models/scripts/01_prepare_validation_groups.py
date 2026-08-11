#!/usr/bin/env python3
"""Prepare vegetation-specific cal/val groups."""

import sys
from pathlib import Path

import pandas as pd
import yaml

# add base directory to sys.path to import train_pipeline.utilsLoading
BASE_DIR_ALL = Path(__file__).parent.parent.parent.parent
sys.path.append(str(BASE_DIR_ALL))

BASE_DIR_ANALYSIS = (
    Path(__file__).resolve().parents[1]
)  # analysis/05_biome_vegetation_specific_models

# from train_pipeline.utilsLoading import load_grounded_eo_validation_data
from train_pipeline.utils_loading import load_grounded_eo_validation_data


def main() -> None:

    # load validation groups
    cols = ["ECO_ID", "NLCD", "uuid", "Network", "Site", "Plot"]
    validation_data = load_grounded_eo_validation_data()[cols]

    # add biome data
    biome_data = pd.read_csv(
        BASE_DIR_ALL / "data/misc/ecoregion_biome_table.csv",
        usecols=["ECO_ID", "BIOME_NAME", "BIOME_NUM", "ECO_NAME"],
    )

    validation_data = validation_data.merge(
        biome_data, how="left", left_on="ECO_ID", right_on="ECO_ID"
    )

    ### Create vegetation-specific groups based on NLCD and BIOME_NAME

    validation_data["NLCD_recoded"] = validation_data["NLCD"].replace(
        {
            "deciduousForest": "Deciduous/Mixed Forest",
            "mixedForest": "Deciduous/Mixed Forest",
            "grasslandHerbaceous": "Grassland",
            "pastureHay": "Grassland",
            "evergreenForest": "Evergreen Forest",
            "shrubScrub": "Shrubland",
            "wetlands": "Wetlands",
            "cultivatedCrops": "Cropland",
        }
    )

    # load recode_table.csv
    recode_table = pd.read_csv(BASE_DIR_ANALYSIS / "data/recode_table.csv")

    # merge recode_table with validation_data on BIOME_NAME and NLCD_recoded
    validation_data = validation_data.merge(
        recode_table,
        how="left",
        left_on=["BIOME_NAME", "NLCD_recoded"],
        right_on=["BIOME_NAME", "NLCD_RECODED"],
    )

    # print (number of NA values in RECODED_GROUP column and number of non-NA values in RECODED_GROUP column)
    print(
        f"Number of NA values in RECODED_GROUP column: {validation_data['RECODED_GROUP'].isna().sum()}"
    )
    print(
        f"Number of non-NA values in RECODED_GROUP column: {validation_data['RECODED_GROUP'].notna().sum()}"
    )
    print(
        f"Number of observations not assigned to group: {(validation_data['RECODED_GROUP'] == -1).sum()}"
    )

    # save recoded validation groups to csv
    validation_data.to_csv(
        BASE_DIR_ANALYSIS / "data/validation_groups.csv", index=False
    )

    # count number of observation, ecoregions and sites per recoded group
    group_counts = (
        validation_data.groupby("RECODED_GROUP")
        .agg(
            n_obs=("uuid", "count"),
            n_ecoregions=("ECO_ID", "nunique"),
            n_sites=("Site", "nunique"),
        )
        .reset_index()
    )
    print(group_counts)

    # save recoded group counts to csv
    group_counts.to_csv(
        BASE_DIR_ANALYSIS / "data/recoded_group_counts.csv", index=False
    )

    # create table: recoded_group, ecoregion_name, n_sites, n_obs: (one row per ecoregion in the recoded group)
    recoded_group_table = (
        validation_data.groupby(["RECODED_GROUP", "ECO_NAME"])
        .agg(
            n_sites=("Site", "nunique"),
            n_obs=("uuid", "count"),
        )
        .reset_index()
    )
    print(recoded_group_table)

    # save recoded table:
    recoded_group_table.to_csv(
        BASE_DIR_ANALYSIS / "data/recoded_group_table.csv", index=False
    )


if __name__ == "__main__":
    main()
