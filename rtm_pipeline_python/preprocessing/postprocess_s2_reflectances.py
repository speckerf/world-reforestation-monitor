import glob
import os

import numpy as np
import pandas as pd
from loguru import logger

from rtm_pipeline_python.utils import load_s2_reflectances

S2_VEGETATION_PATH = (
    "data/rtm_pipeline/output/s2_reflectances/s2_reflectances_vegetation.csv"
)
S2_WATER_PATH = "data/rtm_pipeline/output/s2_reflectances/s2_reflectances_water.csv"
S2_URBAN_PATH = "data/rtm_pipeline/output/s2_reflectances/s2_reflectances_urban.csv"
S2_BARESOIL_PATH = (
    "data/rtm_pipeline/output/s2_reflectances/s2_reflectances_baresoil.csv"
)
S2_SNOWICE_PATH = "data/rtm_pipeline/output/s2_reflectances/s2_reflectances_snowice.csv"

ECO_BIOME_TABLE_PATH = "data/misc/ecoregion_biome_table.csv"


def _save_s2_reflectances_by_lulc():
    df = load_s2_reflectances()

    # save all vegetation spectra (where mask sum equals 0)
    df_vegetation = df[df["sum_masks"] == 0]
    df_vegetation.to_csv(
        S2_VEGETATION_PATH,
        index=False,
    )

    # filter based on sum_masks = 3 : and label in lulc equals water
    # dw: 0
    # proba: 80, 200
    # esa: 80

    df_water = df[
        (df["sum_masks"] == 3)
        & (
            (df["esa_label"] == 80)
            | (df["dw_label"] == 0)
            | (df["proba_label"] == 80)
            | (df["proba_label"] == 200)
        )
    ]

    # save to csv
    df_water.to_csv(
        S2_WATER_PATH,
        index=False,
    )

    # filter based on sum_masks = 3 : and label in lulc equals urban/built/articifial
    # dw: 6
    # proba: 50
    # esa: 50

    df_urban = df[
        (df["sum_masks"] == 3)
        & ((df["esa_label"] == 50) | (df["dw_label"] == 6) | (df["proba_label"] == 50))
    ]

    # save to csv
    df_urban.to_csv(
        S2_URBAN_PATH,
        index=False,
    )

    # filter based on sum_masks = 3 : and label in lulc equals snow/ice
    # dw: 8
    # proba: 70
    # esa: 70

    df_snowice = df[
        (df["sum_masks"] == 3)
        & ((df["esa_label"] == 70) | (df["dw_label"] == 8) | (df["proba_label"] == 70))
    ]

    # save to csv
    df_snowice.to_csv(
        S2_SNOWICE_PATH,
        index=False,
    )

    # filter based on sum_masks = 3 : and label in lulc equals bare soil
    # dw: 7
    # proba: 60
    # esa: 60

    df_baresoil = df[
        (df["sum_masks"] == 3)
        & ((df["esa_label"] == 60) | (df["dw_label"] == 7) | (df["proba_label"] == 60))
    ]

    # save to csv
    df_baresoil.to_csv(
        S2_BARESOIL_PATH,
        index=False,
    )
    return None


def sample_reflectances(data_path, ecoregion, num_samples, int_to_refl=True):
    assert os.path.exists(data_path), f"{data_path} does not exist"
    assert os.path.exists(
        ECO_BIOME_TABLE_PATH
    ), f"{ECO_BIOME_TABLE_PATH} does not exist"

    # Load the data
    data_df = pd.read_csv(data_path)

    # Filter by ecoregion
    filtered_df = data_df[data_df["ECO_ID"] == ecoregion]

    # If the filtered data has fewer samples than required
    if len(filtered_df) < num_samples:

        # Load the ecoregion-biome table
        eco_biome_df = pd.read_csv(ECO_BIOME_TABLE_PATH)

        logger.warning(
            f"Only {len(filtered_df)} samples for {os.path.basename(data_path)} in ecoregion {ecoregion}. Adding samples from the same biome {int(eco_biome_df.loc[eco_biome_df['ECO_ID'] == ecoregion, 'BIOME_NUM'].values[0])}"
        )

        # Get the biome number for the given ecoregion
        biome_num = eco_biome_df.loc[
            eco_biome_df["ECO_ID"] == ecoregion, "BIOME_NUM"
        ].values[0]

        # Filter data by the same biome
        biome_filtered_df = data_df[
            data_df["ECO_ID"].isin(
                eco_biome_df[eco_biome_df["BIOME_NUM"] == biome_num]["ECO_ID"]
            )
        ]

        # Calculate the number of additional samples needed
        additional_samples_needed = num_samples - len(filtered_df)

        # Sample additional rows from the same biome
        additional_samples = biome_filtered_df.sample(
            n=additional_samples_needed, replace=True
        )

        # Combine the original filtered data with the additional samples
        combined_df = pd.concat([filtered_df, additional_samples])
    else:
        # If enough samples, just sample the required number of rows
        combined_df = filtered_df.sample(n=num_samples)

    if int_to_refl:
        # Convert the integer reflectance values to actual reflectance values
        # For columns [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
        refl_columns = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        for col in refl_columns:
            combined_df[col] = combined_df[col] / 10000

    return combined_df


# def load_s2_water_reflectances(eco_id: int):
#     assert os.path.exists(S2_WATER_PATH), f"{S2_WATER_PATH} does not exist"
#     df = pd.read_csv(S2_WATER_PATH)
#     return df[df["ECO_ID"] == eco_id]


# def load_s2_urban_reflectances(eco_id: int):
#     assert os.path.exists(S2_URBAN_PATH), f"{S2_URBAN_PATH} does not exist"
#     return pd.read_csv(S2_URBAN_PATH)


def load_s2_baresoil_reflectances(ecoregion, num_samples):
    return sample_reflectances(S2_BARESOIL_PATH, ecoregion, num_samples)


def load_s2_snowice_reflectances(ecoregion, num_samples):
    return sample_reflectances(S2_SNOWICE_PATH, ecoregion, num_samples)


def load_s2_urban_reflectances(ecoregion, num_samples):
    return sample_reflectances(S2_URBAN_PATH, ecoregion, num_samples)


def load_s2_water_reflectances(ecoregion, num_samples):
    return sample_reflectances(S2_WATER_PATH, ecoregion, num_samples)


def load_s2_vegetation_reflectances(ecoregion, num_samples):
    return sample_reflectances(S2_VEGETATION_PATH, ecoregion, num_samples)


# def load_s2_snowice_reflectances():
#     assert os.path.exists(S2_SNOWICE_PATH), f"{S2_SNOWICE_PATH} does not exist"
#     return pd.read_csv(S2_SNOWICE_PATH)


# def load_s2_vegetation_reflectances():
#     assert os.path.exists(S2_VEGETATION_PATH), f"{S2_VEGETATION_PATH} does not exist"
#     return pd.read_csv(S2_VEGETATION_PATH)


def main():
    # save s2 reflectances for different lulc
    _save_s2_reflectances_by_lulc()
    df_vegetation = load_s2_vegetation_reflectances()
    df_water = load_s2_water_reflectances()
    df_urban = load_s2_urban_reflectances()
    df_baresoil = load_s2_baresoil_reflectances()
    df_snowice = load_s2_snowice_reflectances()

    return None


if __name__ == "__main__":
    main()
