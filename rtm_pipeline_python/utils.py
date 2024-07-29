import glob
import os
import subprocess

import numpy as np
import pandas as pd
import yaml
from loguru import logger


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


def load_baresoil_insitu_multispec(eco_id=None) -> pd.DataFrame:
    raise NotImplementedError
    assert eco_id is None, "Not implemented yet"
    # Load insitu bare soil data
    baresoil_insitu = pd.read_csv(
        os.path.join(
            "..",
            "rtm_simulate_lut",
            "data",
            "soil_spectra_insitu_with_eco_for_prosail.csv",
        )
    )
    return baresoil_insitu


def load_baresoil_insitu_hyperspec(eco_id=None) -> pd.DataFrame:
    raise NotImplementedError
    assert eco_id is None, "Not implemented yet"
    # Load insitu bare soil data
    baresoil_insitu = pd.read_csv(
        os.path.join(
            "..",
            "rtm_simulate_lut",
            "data",
            "soil_spectra_insitu_with_eco_for_prosail.csv",
        )
    )
    return baresoil_insitu


def load_baresoil_emit(eco_id=None) -> pd.DataFrame:
    raise NotImplementedError
    assert eco_id is not None, "Eco ID must be provided"
    # Load insitu bare soil data
    baresoil_emit = pd.read_csv(
        os.path.join(
            "..",
            "rtm_simulate_lut",
            "data",
            "soil_spectra_emit_hyper_with_eco_for_prosail.csv",
        )
    )
    return baresoil_emit


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
            "data/rtm_pipeline/input/s2_reflectances/angles_ecoregion_level/s2_angles_eco_*.csv"
        )
    else:
        all_files = glob.glob(
            f"data/rtm_pipeline/input/s2_reflectances/angles_ecoregion_level/s2_angles_eco_{eco_id}.csv"
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
        return "NULL"
    else:
        return str(value)


def test_yield_hyperparams():
    config_file = "config/rtm_pipeline_hyperparam_opt.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    config_hyperparams = config["hyper_params"]
    config_general = config["general_params"]
    generator = ConfigGenerator(config_hyperparams)

    for config_hyperparam_iter in generator.generate_configs():
        print({"general_params": config_general, "hyperparams": config_hyperparam_iter})


def generate_combinations(d):
    import itertools

    """Recursively generate all combinations of elements in lists within a nested dictionary."""
    if isinstance(d, dict):
        keys, values = zip(*d.items())
        value_combinations = [generate_combinations(v) for v in values]
        for combination in itertools.product(*value_combinations):
            yield dict(zip(keys, combination))
    elif isinstance(d, list):
        for item in d:
            yield from generate_combinations(item)
    else:
        yield d


class ConfigGenerator:
    def __init__(self, config):
        self.config = config

    def generate_configs(self):
        return generate_combinations(self.config)


if __name__ == "__main__":
    # load_s2_reflectances(eco_id=None)
    a = load_s2_angles(eco_id=None)
    b = load_s2_reflectances(eco_id=None)
    print(a.head())

    # test_yield_hyperparams()
