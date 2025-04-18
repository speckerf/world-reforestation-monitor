# Config file parsing
import yaml

with open("config/train_pipeline.yaml", "r") as file:
    config_train_pipeline = yaml.safe_load(file)

with open("config/gee_pipeline.yaml", "r") as file:
    config_gee_pipeline = yaml.safe_load(file)

# this function is used to access the config from anywhere
def get_config(instance: str) -> dict:
    if instance == "gee_pipeline":
        return config_gee_pipeline
    elif instance == "train_pipeline":
        # replace okaceholder in optuna study name with the trait and split and model
        config_train_pipeline["optuna_study_name"] = (
            config_train_pipeline["optuna_study_name"]
            .replace("(TRAIT)", config_train_pipeline["trait"])
            .replace(
                "(SPLIT)", str(config_train_pipeline["group_k_fold_current_split"])
            )
            .replace("(MODEL)", config_train_pipeline["model"])
        )
        return config_train_pipeline
    else:
        raise ValueError(f"Unknown config instance: {instance}")
