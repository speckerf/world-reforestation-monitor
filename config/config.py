# Config file parsing
# Creates class objects from config file, such that config is only loaded and parsed once.

import os

import ee
import yaml
from loguru import logger

# if __name__ == "__main__":
#     ee.Initialize()


class Config:
    def __init__(self, filename: str):
        self.filename = filename
        self.config = self._load_config()

    # allows to call for config['key'] instead of config.config['key']
    def __getitem__(self, key):
        return self.config[key]

    # if printed, print the config
    def __repr__(self):
        return str(self.config)

    def _load_config(self):
        """
        Parse the configuration files for the seed pipeline.
        - gee_pipeline.yaml

        Returns
        -------
        config : dict
            config dictionary
        """
        # check if config folder exists in current directory
        if os.path.exists("config"):
            path_to_root = os.path.join(".")  # current directory
        else:
            # check if config folder exists in parent directory
            if os.path.exists(os.path.join("..", "config")):
                path_to_root = os.path.join("..")  # parent directory
            else:
                raise FileNotFoundError("Config folder not found. ")

        config = self._read_config(os.path.join(path_to_root, "config", self.filename))

        return config

    def _read_config(self, yaml_filepath):
        # Read YAML experiment definition file
        logger.debug(f"Reading config file {yaml_filepath}")
        if not os.path.exists(yaml_filepath):
            raise FileNotFoundError(f"Config file {yaml_filepath} does not exist. ")
        else:
            with open(yaml_filepath, "r") as f:
                config = yaml.safe_load(f)
        return config


# one global config instance
config_gee_pipeline = Config(filename="gee_pipeline.yaml")
config_train_pipeline = Config(filename="train_pipeline.yaml")
config_validation_pipeline = Config(filename="validation_pipeline.yaml")
config_rtm_pipeline = Config(filename="rtm_pipeline.yaml")


# this function is used to access the config from anywhere
def get_config(instance: str) -> Config:
    if instance == "gee_pipeline":
        return config_gee_pipeline
    elif instance == "train_pipeline":
        return config_train_pipeline
    elif instance == "validation_pipeline":
        return config_validation_pipeline
    elif instance == "rtm_pipeline":
        return config_rtm_pipeline
    else:
        raise ValueError(f"Unknown config instance: {instance}")
