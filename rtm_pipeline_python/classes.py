import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import truncnorm

from rtm_pipeline_python.utils import (
    bool_to_r_str,
    int_or_null_to_r_str,
    load_insitu_foliar_generated,
    load_s2_angles,
    predefined_prosail_params,
    rename_angles_utils,
    string_or_null_to_r_str,
)


class rtm_simulator:
    def __init__(self, config, r_script_path):
        self.config = config
        self.prosail_config = predefined_prosail_params(config["parameter_setup"])
        self.r_script_path = r_script_path
        assert os.path.exists(
            self.r_script_path
        ), f"{self.r_script_path} does not exist"
        self.distributions = None
        self.insitu_foliar = None
        self.s2_angles = None
        self.eco_id = None

    def create_distributions(self, params) -> dict:
        distributions = {}

        all_insitu_foliar_params = [
            parm
            for parm, p in params.items()
            if p["distribution"] == "insitu_foliar_codistribution"
        ]
        all_angles_params = [
            parm
            for parm, p in params.items()
            if p["distribution"] == "s2_angles_gee_samples"
        ]
        for parm, p in params.items():
            if p["distribution"] == "truncnorm":
                distributions[parm] = self.get_truncated_normal_sampler(
                    mean=p["mean"], sd=p["std"], low=p["min"], upp=p["max"]
                )
            elif p["distribution"] == "OPTUNA_truncnorm":
                assert f"{parm}_min" in self.config, f"{parm}_min not in config"
                assert f"{parm}_max" in self.config, f"{parm}_max not in config"
                assert f"{parm}_mean" in self.config, f"{parm}_mean not in config"
                assert f"{parm}_std" in self.config, f"{parm}_std not in config"
                distributions[parm] = self.get_truncated_normal_sampler(
                    mean=self.config[f"{parm}_mean"],
                    sd=self.config[f"{parm}_std"],
                    low=self.config[f"{parm}_min"],
                    upp=self.config[f"{parm}_max"],
                )
            elif p["distribution"] == "normal":
                distributions[parm] = self.get_normal_sampler(
                    mean=p["mean"], sd=p["std"]
                )
            elif p["distribution"] == "uniform":
                distributions[parm] = self.get_uniform_sampler(
                    low=p["min"], upp=p["max"]
                )
            elif p["distribution"] == "constant":
                distributions[parm] = self.get_constant_sampler(value=p["value"])
            elif p["distribution"] == "ratio":
                distributions[parm] = self.get_ratio_sampler(
                    ratio=p["ratio"], base_param=p["base_param"]
                )
            elif p["distribution"] == "insitu_foliar_codistribution":
                distributions[parm] = self.get_insitu_foliar_codistribution_sampler(
                    params=all_insitu_foliar_params
                )
            elif p["distribution"] == "s2_angles_gee_samples":
                distributions[parm] = self.get_s2_angles_sampler(
                    params=all_angles_params
                )
        return distributions

    def get_truncated_normal_sampler(self, mean=0, sd=1, low=0, upp=10) -> callable:
        def sampler(size=1):
            return truncnorm(
                (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
            ).rvs(size)

        return sampler

    def get_normal_sampler(self, mean=0, sd=1) -> callable:
        def sampler(size=1):
            return np.random.normal(loc=mean, scale=sd, size=size)

        return sampler

    def get_uniform_sampler(self, low=0, upp=1) -> callable:
        def sampler(size=1):
            return np.random.uniform(low=low, high=upp, size=size)

        return sampler

    def get_constant_sampler(self, value) -> callable:
        def sampler(size=1):
            return np.full(size, value)

        return sampler

    def get_insitu_foliar_codistribution_sampler(self, params: list) -> callable:
        def sampler(size=1) -> pd.DataFrame:
            if size <= len(self.insitu_foliar):
                sampled_values = self.insitu_foliar[params].sample(
                    n=size, replace=False
                )
            else:
                sampled_values = self.insitu_foliar[params].sample(n=size, replace=True)

            return sampled_values

        return sampler

    def get_s2_angles_sampler(self, params: list) -> callable:
        def sampler(size=1) -> pd.DataFrame:
            if size <= len(self.s2_angles):
                return self.s2_angles[params].sample(n=size, replace=False)
            else:
                return self.s2_angles[params].sample(n=size, replace=True)

        return sampler

    def get_ratio_sampler(self, ratio, base_param) -> callable:
        def sampler(size=1, base_values=None):
            if base_values is None:
                raise ValueError(
                    f"Base values for parameter '{base_param}' must be provided for ratio calculation."
                )
            return base_values * ratio

        return sampler

    def generate_prosail_input(self) -> pd.DataFrame:
        self.distributions = self.create_distributions(
            self.prosail_config["prosail_params"]
        )
        self.insitu_foliar = load_insitu_foliar_generated()
        # Generate input reflectances
        number_of_samples = self.config["num_spectra"]

        if self.config["parameter_setup"] == "snap_atbd":
            InputPROSAIL = pd.read_csv(
                "data/rtm_pipeline/input/prosail_atbd/atbd_inputs.csv"
            ).sample(n=number_of_samples)
        else:
            InputPROSAIL = pd.DataFrame(index=range(number_of_samples))

        base_values = {}
        # First, sample all non-ratio parameters
        for parm, sampler in self.distributions.items():
            if self.prosail_config["prosail_params"][parm]["distribution"] in [
                "truncnorm",
                "normal",
                "uniform",
                "constant",
                "OPTUNA_truncnorm",
            ]:
                np.random.seed(42)
                InputPROSAIL[parm] = sampler(size=number_of_samples)
                base_values[parm] = InputPROSAIL[parm]

        # if insitu foliar parameters are needed, sample them, need to be sampled for whole rows in insitu foliar data

        # Sample in-situ foliar data for all traits together
        insitu_foliar_sample = None
        for parm, sampler in self.distributions.items():
            if (
                self.prosail_config["prosail_params"][parm]["distribution"]
                == "insitu_foliar_codistribution"
            ):
                if insitu_foliar_sample is None:
                    insitu_foliar_sample = sampler(size=number_of_samples)
                InputPROSAIL[parm] = insitu_foliar_sample[parm].values
                base_values[parm] = InputPROSAIL[parm]

        # sample s2 angles for all parameters that need them
        s2_angles_sample = None
        for parm, sampler in self.distributions.items():
            if (
                self.prosail_config["prosail_params"][parm]["distribution"]
                == "s2_angles_gee_samples"
            ):
                if s2_angles_sample is None:
                    s2_angles_sample = sampler(size=number_of_samples)
                InputPROSAIL[parm] = s2_angles_sample[parm].values

        # Then, handle the ratio-based and insitu foliar parameters
        for parm, sampler in self.distributions.items():
            if self.prosail_config["prosail_params"][parm]["distribution"] == "ratio":
                # ratio = self.config["prosail_params"][parm]["ratio"]
                base_param = self.prosail_config["prosail_params"][parm]["base_param"]
                InputPROSAIL[parm] = sampler(
                    size=number_of_samples, base_values=base_values[base_param]
                )

        if "EWT" not in InputPROSAIL.columns:
            assert (
                "Cw_rel" in InputPROSAIL.columns
            ), f"Cw_rel not in InputPROSAIL.columns"
            # InputPROSAIL$EWT <- ((InputPROSAIL$LMA)/(1-InputPROSAIL$Cw_rel))-InputPROSAIL$LMA
            InputPROSAIL["EWT"] = (
                InputPROSAIL["LMA"] / (1 - InputPROSAIL["Cw_rel"])
            ) - InputPROSAIL["LMA"]
            # remove column Cw_rel
            InputPROSAIL = InputPROSAIL.drop(columns=["Cw_rel"])

        return InputPROSAIL

    def call_prosail(self, InputPROSAIL: pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "InputPROSAIL.csv")
            output_path = os.path.join(temp_dir, "OutputPROSAIL.csv")

            # Save InputPROSAIL to CSV
            InputPROSAIL.to_csv(input_path, index=False)

            # Prepare noise arguments
            noise_bool = True
            noise_type = "addmulti"
            if noise_type == "addmulti":
                noise_args = {
                    "AdditiveNoise": self.config["additive_noise"],
                    "MultiplicativeNoise": self.config["multiplicative_noise"],
                }
                noise_args = json.dumps(noise_args)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            # Prepare rsoil arguments
            modify_rsoil = True
            rsoil_insitu = (
                True if self.config.get("rsoil_emit_insitu", "") == "insitu" else False
            )
            rsoil_emit = (
                True if self.config.get("rsoil_emit_insitu", "") == "emit" else False
            )
            rsoil_insitu_fraction = self.config["rsoil_fraction"] if rsoil_insitu else 0
            rsoil_emit_fraction = self.config["rsoil_fraction"] if rsoil_emit else 0

            process = subprocess.Popen(
                [
                    "/usr/local/bin/Rscript",
                    self.r_script_path,
                    "--input",
                    input_path,
                    "--output",
                    output_path,
                    "--add_noise",
                    bool_to_r_str(noise_bool),
                    "--noise_type",
                    string_or_null_to_r_str(noise_type),
                    "--noise_args",
                    noise_args,
                    "--ecoregion",
                    int_or_null_to_r_str(self.eco_id),
                    "--modify_rsoil",
                    bool_to_r_str(modify_rsoil),
                    "--rsoil_insitu",
                    bool_to_r_str(rsoil_insitu),
                    "--rsoil_insitu_fraction",
                    str(rsoil_insitu_fraction),
                    "--rsoil_emit",
                    bool_to_r_str(rsoil_emit),
                    "--rsoil_emit_fraction",
                    str(rsoil_emit_fraction),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Read and print output and error streams in real-time
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output:
                    print(output.strip())
                if error:
                    print(f"{error.strip()}")

                # Break loop if process is done and both streams are empty
                if output == "" and error == "" and process.poll() is not None:
                    break

            rc = process.poll()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, process.args)
            else:
                logger.debug("PROSAIL simulation completed successfully.")

            # Load OutputPROSAIL from CSV
            OutputPROSAIL = pd.read_csv(output_path)

        return OutputPROSAIL

    def _load_s2_lulc_reflectances(
        self, function: callable, num_samples: int, eco_id=None
    ) -> pd.DataFrame:
        df = function(ecoregion=eco_id, num_samples=num_samples)
        df_return = rename_angles_utils(df)
        return df_return

    def generate_lut(self, eco_id=None):

        self.eco_id = eco_id
        logger.info(f"Generating LUT for ecoregion {eco_id}.")

        self.s2_angles = load_s2_angles(eco_id=self.eco_id, resync=False)
        # Generate input reflectances
        InputPROSAIL = self.generate_prosail_input()

        # Run PROSAIL
        OutputPROSAIL = self.call_prosail(InputPROSAIL)

        return OutputPROSAIL


def prepare_dataset(df: pd.DataFrame):
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    if "solar_azimuth" in df.columns:

        df["relative_azimuth"] = np.abs(df["solar_azimuth"] - df["view_azimuth"])

        # rename columns to match the expected names
        df = df.rename(
            columns={
                "solar_zenith": "tts",
                "view_zenith": "tto",
                "relative_azimuth": "psi",
            }
        )
        angles = ["tts", "tto", "psi"]

        # drop the original columns
        df = df.drop(columns=["solar_azimuth", "view_azimuth"])
    else:
        angles = ["tts", "tto", "psi"]
        assert all(
            [angle in df.columns for angle in angles]
        ), f"Missing angle columns: {angles}"

    # if bands are integers and larger than 1, divide by 10000
    if all([df[band].dtype == int and df[band].max() > 1 for band in bands]):
        logger.debug("Dividing bands by 10000")
        df[bands] = df[bands] / 10000

    return df[[*bands, *angles]]


def get_baresoil_s2(n=10) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "rtm_pipeline",
        "output",
        "s2_reflectances",
        "s2_reflectances_baresoil.csv",
    )
    df = pd.read_csv(path).sample(n=n)
    return prepare_dataset(df)


def get_water_s2(n=10) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "rtm_pipeline",
        "output",
        "s2_reflectances",
        "s2_reflectances_water.csv",
    )
    df = pd.read_csv(path).sample(n=n)
    return prepare_dataset(df)


def get_urban_s2(n=10) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "rtm_pipeline",
        "output",
        "s2_reflectances",
        "s2_reflectances_urban.csv",
    )
    df = pd.read_csv(path).sample(n=n)
    return prepare_dataset(df)


def get_snowice_s2(n=10) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "rtm_pipeline",
        "output",
        "s2_reflectances",
        "s2_reflectances_snowice.csv",
    )
    df = pd.read_csv(path).sample(n=n)
    return prepare_dataset(df)


def get_baresoil_insitu(n=10) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "rtm_pipeline",
        "output",
        "insitu_soil_database",
        "insitu_soil_spectra_sentinel2bands.csv",
    )
    df = pd.read_csv(path).sample(n=n)

    # select only the bands and set angles to 0
    df = df[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
    df["tts"] = 0
    df["tto"] = 0
    df["psi"] = 0

    return df


def get_baresoil_emit(n=10) -> pd.DataFrame:
    path = os.path.join(
        "data",
        "rtm_pipeline",
        "output",
        "emit_hyperspectral",
        "point_data",
        "global-baresoil-random-points-all_sentinel2bands.csv",
    )
    df = pd.read_csv(path).sample(n=n)

    # select only the bands and set angles to 0
    df = df[["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]]
    df["tts"] = 0
    df["tto"] = 0
    df["psi"] = 0

    return df


def helper_apply_posthoc_modifications(single_df, trait, config):
    single_df = apply_posthoc_modifications(single_df, trait, config)
    # set all negative reflectances to 0 / in columns B2 - B12
    single_df.loc[:, "B2":"B12"] = single_df.loc[:, "B2":"B12"].clip(lower=0)
    return single_df


def apply_posthoc_modifications(df: pd.DataFrame, trait: str, config: dict):
    angles = ["tts", "tto", "psi"]

    df_baresoil_insitu = get_baresoil_insitu(
        n=int(len(df) * config["p_baresoil_insitu"])
    )
    df_baresoil_s2 = get_baresoil_s2(n=int(len(df) * config["p_baresoil_s2"]))
    df_baresoil_emit = get_baresoil_emit(n=int(len(df) * config["p_baresoil_emit"]))
    df_urban_s2 = get_urban_s2(n=int(len(df) * config["p_urban_s2"]))
    df_snowice_s2 = get_snowice_s2(n=int(len(df) * config["p_snowice_s2"]))

    df_posthoc_additions = pd.concat(
        [
            df_baresoil_insitu,
            df_baresoil_s2,
            df_baresoil_emit,
            df_urban_s2,
            df_snowice_s2,
        ],
        ignore_index=True,
    )

    # add angles to the posthoc additions
    df_posthoc_additions[angles] = (
        df[angles].sample(n=len(df_posthoc_additions), replace=True).values
    )

    # set trait to zero
    df_posthoc_additions[trait] = 0

    # # add small random positive noise to the added trait values
    # df_posthoc_additions[trait] = df_posthoc_additions[trait] + np.abs(
    #     np.random.rand(len(df_posthoc_additions)) * 1e-3
    # )

    # add the posthoc additions to the original dataframe
    df = pd.concat([df, df_posthoc_additions], ignore_index=True)

    # Set all negative reflectances to 0 / in columns B2 - B12
    df.loc[:, "B2":"B12"] = df.loc[:, "B2":"B12"].clip(lower=0)

    return df


if __name__ == "__main__":
    pass
