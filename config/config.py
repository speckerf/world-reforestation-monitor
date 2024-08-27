# Config file parsing
import yaml

# with open("config/rtm_pipeline.yaml", "r") as file:
#     config_rtm_pipeline = yaml.safe_load(file)

with open("config/train_pipeline.yaml", "r") as file:
    config_train_pipeline = yaml.safe_load(file)

with open("config/validation_pipeline.yaml", "r") as file:
    config_validation_pipeline = yaml.safe_load(file)

with open("config/gee_pipeline.yaml", "r") as file:
    config_gee_pipeline = yaml.safe_load(file)

with open("config/ecoregions_simplify_export.yaml", "r") as file:
    config_eco_simple = yaml.safe_load(file)


# this function is used to access the config from anywhere
def get_config(instance: str) -> dict:
    if instance == "gee_pipeline":
        return config_gee_pipeline
    elif instance == "train_pipeline":
        return config_train_pipeline
    elif instance == "validation_pipeline":
        return config_validation_pipeline
    elif instance == "ecoregions_simple":
        return config_eco_simple
    # elif instance == "rtm_pipeline":
    #     return config_rtm_pipeline
    else:
        raise ValueError(f"Unknown config instance: {instance}")
