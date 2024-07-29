import glob
import json
import os
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import get_config
from rtm_pipeline_python.preprocessing.postprocess_s2_reflectances import (  # load_s2_reflectances,; load_s2_vegetation_reflectances,
    load_s2_baresoil_reflectances,
    load_s2_snowice_reflectances,
    load_s2_urban_reflectances,
    load_s2_water_reflectances,
)
from rtm_pipeline_python.utils import (
    bool_to_r_str,
    int_or_null_to_r_str,
    load_insitu_foliar,
    load_s2_angles,
    rename_angles_utils,
)


class rtm_simulator:
    def __init__(self, config, r_script_path):
        self.config = config
        self.r_script_path = r_script_path
        assert os.path.exists(
            self.r_script_path
        ), f"{self.r_script_path} does not exist"
        self.prospect_version = "..."
        self.four_sail_version = "..."
        self.distributions = self.create_distributions(config["prosail_params"])
        self.insitu_foliar = load_insitu_foliar()
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
        # Generate input reflectances
        number_of_samples = self.config["number_of_samples"]

        # init InputPROSAIL with number of samples
        InputPROSAIL = pd.DataFrame(index=range(number_of_samples))

        base_values = {}
        # First, sample all non-ratio parameters
        for parm, sampler in self.distributions.items():
            if self.config["prosail_params"][parm]["distribution"] in [
                "truncnorm",
                "normal",
                "uniform",
                "constant",
            ]:
                np.random.seed(42)
                InputPROSAIL[parm] = sampler(size=number_of_samples)
                base_values[parm] = InputPROSAIL[parm]

        # if insitu foliar parameters are needed, sample them, need to be sampled for whole rows in insitu foliar data

        # Sample in-situ foliar data for all traits together
        insitu_foliar_sample = None
        for parm, sampler in self.distributions.items():
            if (
                self.config["prosail_params"][parm]["distribution"]
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
                self.config["prosail_params"][parm]["distribution"]
                == "s2_angles_gee_samples"
            ):
                if s2_angles_sample is None:
                    s2_angles_sample = sampler(size=number_of_samples)
                InputPROSAIL[parm] = s2_angles_sample[parm].values

        # Then, handle the ratio-based and insitu foliar parameters
        for parm, sampler in self.distributions.items():
            if self.config["prosail_params"][parm]["distribution"] == "ratio":
                # ratio = self.config["prosail_params"][parm]["ratio"]
                base_param = self.config["prosail_params"][parm]["base_param"]
                InputPROSAIL[parm] = sampler(
                    size=number_of_samples, base_values=base_values[base_param]
                )

        return InputPROSAIL

    def call_prosail(self, InputPROSAIL: pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "InputPROSAIL.csv")
            output_path = os.path.join(temp_dir, "OutputPROSAIL.csv")

            # Save InputPROSAIL to CSV
            InputPROSAIL.to_csv(input_path, index=False)

            # Prepare noise arguments
            noise_bool = self.config["add_noise"]
            noise_type = self.config["lut_forward_params"]["add_noise"]["noise_type"]
            noise_args = self.config["lut_forward_params"]["add_noise"].get(
                "arguments", {}
            )

            # Prepare rsoil arguments
            modify_rsoil = self.config["modify_rsoil"]
            rsoil_insitu = self.config["lut_forward_params"]["modify_rsoil"][
                "rsoil_from_insitu"
            ]["bool"]
            rsoil_insitu_fraction = self.config["lut_forward_params"]["modify_rsoil"][
                "rsoil_from_insitu"
            ]["fraction"]
            rsoil_emit = self.config["lut_forward_params"]["modify_rsoil"][
                "rsoil_from_emit"
            ]["bool"]
            rsoil_emit_fraction = self.config["lut_forward_params"]["modify_rsoil"][
                "rsoil_from_emit"
            ]["fraction"]

            # Ensure noise_args is always a valid JSON string
            if not noise_args:
                noise_args = "{}"
            else:
                noise_args = json.dumps(noise_args)

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
                    noise_type,
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

    def apply_posthoc_modifications(
        self, lut: pd.DataFrame, eco_id=None
    ) -> pd.DataFrame:

        # if self.config["add_s2_baresoil_spectra"]:
        #     baresoil = self._load_s2_lulc_reflectances(
        #         load_s2_baresoil_reflectances,
        #         self.config["lut_posthoc_params"]["add_s2_baresoil_spectra"][
        #             "num_spectra"
        #         ],
        #         eco_id=eco_id,
        #     )
        #     lut = pd.concat([lut, baresoil])

        # if self.config["add_s2_urban_spectra"]:
        #     urban = self._load_s2_lulc_reflectances(
        #         load_s2_urban_reflectances,
        #         self.config["lut_posthoc_params"]["add_s2_urban_spectra"][
        #             "num_spectra"
        #         ],
        #         eco_id=eco_id,
        #     )
        #     lut = pd.concat([lut, urban])

        # if self.config["add_s2_water_spectra"]:
        #     water = self._load_s2_lulc_reflectances(
        #         load_s2_water_reflectances,
        #         self.config["lut_posthoc_params"]["add_s2_water_spectra"][
        #             "num_spectra"
        #         ],
        #         eco_id=eco_id,
        #     )
        #     lut = pd.concat([lut, water])

        # if self.config["add_s2_snow_spectra"]:
        #     snow = self._load_s2_lulc_reflectances(
        #         load_s2_snowice_reflectances,
        #         self.config["lut_posthoc_params"]["add_s2_snow_spectra"]["num_spectra"],
        #         eco_id=eco_id,
        #     )
        #     lut = pd.concat([lut, snow])

        if self.config["add_s2_nonvegetated_spectra"]:
            baresoil = self._load_s2_lulc_reflectances(
                load_s2_baresoil_reflectances,
                self.config["add_s2_baresoil_spectra"]["num_spectra"],
                eco_id=eco_id,
            )
            urban = self._load_s2_lulc_reflectances(
                load_s2_urban_reflectances,
                self.config["add_s2_urban_spectra"]["num_spectra"],
                eco_id=eco_id,
            )
            water = self._load_s2_lulc_reflectances(
                load_s2_water_reflectances,
                self.config["add_s2_water_spectra"]["num_spectra"],
                eco_id=eco_id,
            )
            snow = self._load_s2_lulc_reflectances(
                load_s2_snowice_reflectances,
                self.config["add_s2_snow_spectra"]["num_spectra"],
                eco_id=eco_id,
            )
            lut = pd.concat([lut, baresoil, urban, water, snow])

        # replace nan with 0 for all trait columns
        for trait in self.config["traits"]:
            lut[trait] = lut[trait].fillna(0)

        # drop unnecessary columns
        columns_to_drop = [
            "dw_label",
            "proba_label",
            "esa_label",
            "dw_mask",
            "proba_mask",
            "esa_mask",
            "sum_masks",
        ]
        if all(col in lut.columns for col in columns_to_drop):
            lut = lut.drop(columns=columns_to_drop)

        return lut

    def generate_lut(self, eco_id=None):
        if self.config["lut_per_ecoregion"] is False and eco_id is not None:
            logger.warning(
                "The 'lut_per_ecoregion' parameter is set to False. The 'eco_id' parameter will be ignored."
            )
            self.eco_id = None
        else:
            self.eco_id = eco_id
            logger.info(f"Generating LUT for ecoregion {eco_id}.")

        self.s2_angles = load_s2_angles(eco_id=self.eco_id, resync=False)
        # Generate input reflectances
        InputPROSAIL = self.generate_prosail_input()

        # Run PROSAIL
        OutputPROSAIL = self.call_prosail(InputPROSAIL)

        # PostHoc modifications
        OutputPROSAIL_POSTHOC = self.apply_posthoc_modifications(
            OutputPROSAIL, eco_id=self.eco_id
        )

        return OutputPROSAIL_POSTHOC


if __name__ == "__main__":
    pass
