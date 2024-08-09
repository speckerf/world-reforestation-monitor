import glob
import os
import subprocess

import numpy as np
import pandas as pd
import yaml
from loguru import logger


def predefined_prosail_params(descriptor: str) -> dict:
    assert descriptor in [
        "wan_2024_lai",
        "snap_atbd",
        "foliar_codistribution",
        "kovacs_2023",
        "estevez_2022",
        "wan_2024_chl",
        "custom_ewt",
        "custom_lma",
    ], f"Unknown descriptor: {descriptor}"

    if descriptor == "wan_2024_lai":
        with open(
            os.path.join("config", "rtm_simulator", "wan_2024_lai.yaml"),
            "r",
        ) as file:
            prosail_config = yaml.safe_load(file)
    elif descriptor == "snap_atbd":
        with open(
            os.path.join("config", "rtm_simulator", "snap_atbd.yaml"), "r"
        ) as file:
            prosail_config = yaml.safe_load(file)
    elif descriptor == "foliar_codistribution":
        with open(
            os.path.join("config", "rtm_simulator", "foliar_codistribution.yaml"),
            "r",
        ) as file:
            prosail_config = yaml.safe_load(file)
    elif descriptor == "kovacs_2023":
        with open(
            os.path.join("config", "rtm_simulator", "kovacs_2023.yaml"),
            "r",
        ) as file:
            prosail_config = yaml.safe_load(file)
    elif descriptor == "estevez_2022":
        with open(
            os.path.join("config", "rtm_simulator", "estevez_2022.yaml"),
            "r",
        ) as file:
            prosail_config = yaml.safe_load(file)
    return prosail_config


def load_insitu_foliar() -> pd.DataFrame:
    # Load insitu foliar data
    foliar_generated = (
        pd.read_csv(
            os.path.join(
                "data",
                "validation_pipeline",
                "output",
                "EXPORT_NEON_foliar_reflectances_with_angles.csv",
            )
        )
        .rename(
            columns={
                "ewt_cm": "EWT",
                "leafMassPerArea_g_cm2": "LMA",
                "carotenoid_mug_cm2": "CAR",
                "chlorophyll_ab_mug_cm2": "CHL",
            }
        )
        .drop(columns=["system:index"])
    )

    foliar_generated["Cw_rel"] = 1 - (
        foliar_generated["LMA"] / (foliar_generated["LMA"] + foliar_generated["EWT"])
    )

    return foliar_generated


def load_s2_angles(eco_id=None, resync=False) -> pd.DataFrame:
    # syn folder from google cloud storage: felixspecker/open-earth/s2_reflectances/ecoregion_level_all_lc/
    # to local repository: data/s2_reflectances/ecoregion_level_all_lc
    # use gsutil -m rsync -d -r gs://felixspecker/open-earth/s2_reflectances/ecoregion_level_all_lc data/s2_reflectances/ecoregion_level_all_lc
    if resync:
        subprocess.run(
            [
                "gcloud",
                "storage",
                "rsync",
                "-r",
                "gs://felixspecker/open-earth/s2_reflectances/angles_ecoregion_level",
                "data/rtm_pipeline/input/s2_reflectances/angles_ecoregion_level",
            ],
            check=True,
        )

    if eco_id is None:

        all_files = glob.glob(
            os.path.join(
                "data/rtm_pipeline/input/s2_reflectances/angles_ecoregion_level/s2_angles_eco_*.csv",
            )
        )
    else:
        all_files = glob.glob(
            os.path.join(
                f"data/rtm_pipeline/input/s2_reflectances/angles_ecoregion_level/s2_angles_eco_{eco_id}.csv",
            )
        )
    all_dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file, index_col="system:index")
            df["ECO_ID"] = int(file.split("_")[-1].split(".")[0])
            # drop column '.geo'
            df = df.drop(columns=[".geo"])
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
    s2_angles = pd.concat(all_dfs, ignore_index=True)

    # convert to tts, tto, psi: rename columns
    s2_angles["psi"] = np.abs(s2_angles["view_azimuth"] - s2_angles["solar_azimuth"])
    s2_angles = s2_angles.drop(columns=["view_azimuth", "solar_azimuth"])
    s2_angles = s2_angles.rename(
        columns={
            "view_zenith": "tto",
            "solar_zenith": "tts",
        }
    )
    return s2_angles


def load_s2_reflectances(eco_id=None, resync=False) -> pd.DataFrame:
    # syn folder from google cloud storage: felixspecker/open-earth/s2_reflectances/ecoregion_level_all_lc/
    # to local repository: data/s2_reflectances/ecoregion_level_all_lc
    # use gsutil -m rsync -d -r gs://felixspecker/open-earth/s2_reflectances/ecoregion_level_all_lc data/s2_reflectances/ecoregion_level_all_lc
    if resync:
        subprocess.run(
            [
                "gcloud",
                "storage",
                "rsync",
                "-r",
                "gs://felixspecker/open-earth/s2_reflectances/reflectances_angles_ecoregion_level_with_lulc",
                "data/rtm_pipeline/input/s2_reflectances/reflectances_angles_ecoregion_level_with_lulc",
            ],
            check=True,
        )

    if eco_id is None:
        # list all files in the folder with the following filename: 's2_reflectances_100000_eco_*.csv'
        # load all files into a single dataframe
        all_files = glob.glob(
            "data/rtm_pipeline/input/s2_reflectances/reflectances_angles_ecoregion_level_with_lulc/s2_reflectances_10000_eco_*.csv"
        )
        # load all files into a list of dataframes, omit files that raise errors
        all_dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file, nrows=10000, index_col="system:index")
                df["ECO_ID"] = int(file.split("_")[-1].split(".")[0])
                # drop column '.geo'
                df = df.drop(columns=[".geo", "random"])
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    else:
        # check if the file with the following filename exists: 's2_reflectances_100000_eco_{eco_id}.csv'
        file = f"data/rtm_pipeline/input/s2_reflectances/ecoregion_level_all_lc/s2_reflectances_10000_eco_{eco_id}.csv"
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"File 's2_reflectances_10000_eco_{eco_id}.csv' not found."
            )
        # load the file into a dataframe
        try:
            df = pd.read_csv(file, nrows=10000, index_col="system:index")
            df["ECO_ID"] = int(file.split("_")[-1].split(".")[0])
            df = df.drop(columns=[".geo", "random"])
        except Exception as e:
            logger.error(f"Error loading file {file}: {e}")
        return df


def rename_angles_utils(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the columns of the angle columns to match the PROSAIL model

    Description:
    ------------
    The PROSAIL model expects the following angle names:
    - 'tts': Solar zenith angle
    - 'tto': View zenith angle
    - 'psi': Relative azimuth angle

    The input dataframes can have different column names for these angles.
    This function renames the columns to match the PROSAIL model.

    psi: 'relative_azimuth'
    tts: 'solar_zenith'
    tto: 'view_zenith'

    or contain view_azimuth and solar_azimuth columns instead of relative_azimuth, then we need to calculate psi from these two columns: psi = abs(solar_azimuth - view_azimuth)
    """

    if "relative_azimuth" in df.columns:
        assert all(
            [
                col in df.columns
                for col in ["solar_zenith", "view_zenith", "relative_azimuth"]
            ]
        )
        df = df.rename(
            columns={
                "relative_azimuth": "psi",
                "solar_zenith": "tts",
                "view_zenith": "tto",
            }
        )
    elif "solar_azimuth" in df.columns and "view_azimuth" in df.columns:
        assert all(
            [
                col in df.columns
                for col in [
                    "solar_zenith",
                    "view_zenith",
                    "solar_azimuth",
                    "view_azimuth",
                ]
            ]
        )
        df["psi"] = abs(df["solar_azimuth"] - df["view_azimuth"])
        df = df.rename(
            columns={
                "solar_zenith": "tts",
                "view_zenith": "tto",
            }
        )
        df = df.drop(columns=["solar_azimuth", "view_azimuth"])

    elif "tts" in df.columns and "tto" in df.columns and "psi" in df.columns:
        pass

    else:
        raise ValueError(
            "The input dataframe does not contain the expected angle columns."
        )

    return df


def bool_to_r_str(value):
    assert isinstance(value, bool)
    return "TRUE" if value else "FALSE"


def int_or_null_to_r_str(value):
    assert value is None or isinstance(value, int)
    if value is None:
        return "All"
    else:
        return str(value)


def string_or_null_to_r_str(value):
    assert value is None or isinstance(value, str)
    if value is None:
        return "NULL"
    else:
        return value
