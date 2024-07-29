import os
import random

import yaml
from loguru import logger

from rtm_pipeline_python.classes import rtm_simulator
from rtm_pipeline_python.utils import ConfigGenerator


def create_luts_wrapper():
    config_file = "config/rtm_pipeline_hyperparam_opt.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_(file)

    general_params = config["general_params"]
    hyper_params = config["hyper_params"]
    config_generator = ConfigGenerator(hyper_params)

    random_experiment_folder_name = "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz", k=5)
    )
    for i, current_config in enumerate(config_generator.generate_configs()):
        logger.debug(f"Generating LUT for config {i+1}")
        create_lut(
            config={**general_params, **current_config},
            basefolder=random_experiment_folder_name,
        )


def create_lut(config=None, basefolder=None):
    if config is None:
        config_file = "config/rtm_pipeline.yaml"
        logger.debug(f"Reading config from {config_file}")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        logger.debug("Using provided config.")

    simulator = rtm_simulator(
        config,
        r_script_path=os.path.join(
            "rtm_pipeline_R",
            "src",
            "run_prosail.R",
        ),
    )

    # save luts to csv: create folder with random name in data/rtm_pipeline/output/luts/{random_folder_name} and save rtm_pipeline.yaml in that folder with name rtm_pipeline_{random_folder_name}.yaml
    random_folder_name = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))

    if basefolder is not None:
        lut_folder = os.path.join(
            "data",
            "rtm_pipeline",
            "output",
            "luts",
            "test_ecos",
            basefolder,
            random_folder_name,
        )
    else:
        lut_folder = os.path.join(
            "data", "rtm_pipeline", "output", "luts", "test_ecos", random_folder_name
        )
    os.makedirs(lut_folder, exist_ok=True)

    if config["lut_per_ecoregion"]:
        # create lut per eco using dict comprehension
        luts = {
            eco_id: simulator.generate_lut(eco_id=eco_id)
            for eco_id in config["list_ecoids_in_lai_validation"]
        }
    else:
        luts = {"all_ecoregions": simulator.generate_lut(eco_id=None)}

    for eco_id, lut in luts.items():
        lut.to_csv(os.path.join(lut_folder, f"lut_{eco_id}.csv"), index=False)
    with open(
        os.path.join(lut_folder, f"rtm_pipeline_{random_folder_name}.yaml"), "w"
    ) as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    create_luts_wrapper()
