import numpy as np
from loguru import logger

from config.config import get_config


def log_splits(splits, df_val_trait, current_fold=None):
    logger.trace("Number of splits: {len(splits)}")
    if current_fold is None:
        logger.trace(
            f"Number of observations in train splits: {[len(split[0]) for split in splits]}"
        )
        logger.trace(
            f"Number of observations in test splits: {[len(split[1]) for split in splits]}"
        )
        logger.trace(
            f"Number of unique ECO_IDs in train splits: {[len(np.unique(df_val_trait['ECO_ID'].values[split[0]])) for split in splits]}"
        )
        logger.trace(
            f"Number of unique ECO_IDs in test splits: {[len(np.unique(df_val_trait['ECO_ID'].values[split[1]])) for split in splits]}"
        )
    if current_fold is not None:
        logger.trace(f"Current fold: {current_fold}")
        logger.trace(
            f"Current eco_id in train split: {np.unique(df_val_trait['ECO_ID'].values[splits[current_fold][0]])}"
        )
        logger.trace(
            f"Current eco_id in test split: {np.unique(df_val_trait['ECO_ID'].values[splits[current_fold][1]])}"
        )
        logger.trace(
            f"Current number of observations in train split: {len(splits[current_fold][0])}"
        )
        logger.trace(
            f"Current number of observations in test split: {len(splits[current_fold][1])}"
        )


def merge_dicts_safe(*dicts):
    merged_dict = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                logger.error(f"Duplicate key found: {key}")
                raise ValueError(f"Duplicate key found: {key}")
            merged_dict[key] = value
    return merged_dict


def optuna_init_config(trial):
    config_training = get_config("train_pipeline")
    trait = config_training["trait"]

    if trait == "fapar":
        config_transform_target = {
            "transform_target": trial.suggest_categorical(
                "transform_target", ["logit", "None"]
            ),
        }
    elif trait == "fcover":
        config_transform_target = {
            "transform_target": trial.suggest_categorical(
                "transform_target", ["logit", "None"]
            ),
        }
    elif trait == "lai":
        config_transform_target = {
            "transform_target": trial.suggest_categorical(
                "transform_target", ["log1p", "standard", "None"]
            ),
        }
    else:
        config_transform_target = {
            "transform_target": trial.suggest_categorical(
                "transform_target", ["log1p", "standard", "None"]
            ),
        }

    config_general = {
        "use_angles_for_prediction": trial.suggest_categorical(
            "use_angles_for_prediction", [True, False]
        ),
        # "use_angles_for_prediction": False,
        "posthoc_modifications": trial.suggest_categorical(
            "posthoc_modifications", [True, False]
        ),
        **config_transform_target,
        "nirv_norm": trial.suggest_categorical("nirv_norm", [True, False]),
        # "nirv_norm": True,
    }

    config_lut = {
        "modify_rsoil": trial.suggest_categorical("modify_rsoil", [True, False]),
        # "modify_rsoil": False,
        "add_noise": trial.suggest_categorical("add_noise", [True, False]),
        # "add_noise": True,
        "num_spectra": 2500 * trial.suggest_int("num_spectra_optuna", 1, 20, log=True),
        # "num_spectra": 1000,
        "parameter_setup": trial.suggest_categorical(
            "parameter_setup",
            [
                "estevez_2022",
                "foliar_codistribution",
                "kovacs_2023",
                "snap_atbd",
                "wan_2024_lai",
            ],
        ),
        # "parameter_setup": "foliar_codistribution",
    }
    if config_lut["add_noise"]:
        config_lut["noise_type"] = trial.suggest_categorical(
            "noise_type", ["atbd", "addmulti"]
        )
        if config_lut["noise_type"] == "addmulti":
            config_lut["additive_noise"] = 0.005 * trial.suggest_int(
                "additive_noise_optuna", 1, 6
            )
            config_lut["multiplicative_noise"] = 0.01 * trial.suggest_int(
                "multiplicative_noise_optuna)", 1, 10
            )

    if config_lut["modify_rsoil"]:
        config_lut["rsoil_emit_insitu"] = trial.suggest_categorical(
            "rsoil_emit_insitu", ["emit", "insitu"]
        )
        config_lut["rsoil_fraction"] = 0.1 * trial.suggest_int(
            "rsoil_fraction_optuna", 1, 10
        )

    if config_general["posthoc_modifications"]:
        config_posthoc = {
            "use_baresoil_insitu": trial.suggest_categorical(
                "use_baresoil_insitu", [True, False]
            ),
            "use_baresoil_s2": trial.suggest_categorical(
                "use_baresoil_s2", [True, False]
            ),
            "use_urban_s2": trial.suggest_categorical("use_urban_s2", [True, False]),
            "use_water_s2": trial.suggest_categorical("use_water_s2", [True, False]),
            "use_snowice_s2": trial.suggest_categorical(
                "use_snowice_s2", [True, False]
            ),
            "use_baresoil_emit": trial.suggest_categorical(
                "use_baresoil_emit", [True, False]
            ),
        }

        if config_posthoc["use_baresoil_insitu"]:
            config_posthoc["n_baresoil_insitu"] = 20 * trial.suggest_int(
                "n_baresoil_insitu_optuna", 1, 50, log=True
            )
        if config_posthoc["use_baresoil_s2"]:
            config_posthoc["n_baresoil_s2"] = 20 * trial.suggest_int(
                "n_baresoil_s2_optuna", 1, 50, log=True
            )
        if config_posthoc["use_urban_s2"]:
            config_posthoc["n_urban_s2"] = 20 * trial.suggest_int(
                "n_urban_s2_optuna", 1, 25, log=True
            )
        if config_posthoc["use_water_s2"]:
            config_posthoc["n_water_s2"] = 20 * trial.suggest_int(
                "n_water_s2_optuna", 1, 25, log=True
            )
        if config_posthoc["use_snowice_s2"]:
            config_posthoc["n_snowice_s2"] = 20 * trial.suggest_int(
                "n_snowice_s2_optuna", 1, 25, log=True
            )
        if config_posthoc["use_baresoil_emit"]:
            config_posthoc["n_baresoil_emit"] = 20 * trial.suggest_int(
                "n_baresoil_emit_optuna", 1, 50, log=True
            )

    else:
        config_posthoc = {}

    if config_training["model"] == "mlp":
        hidden_layers_dict = {
            "5": (5,),
            "10": (10,),
            "5_5": (5, 5),
            "5_10": (5, 10),
            "10_5": (10, 5),
            "10_10": (10, 10),
            "5_5_5": (5, 5, 5),
        }

        config_ml = {
            "hidden_layers_optuna": trial.suggest_categorical(
                "hidden_layers",
                [
                    "5",
                    "10",
                    "5_5",
                    "5_10",
                    "10_5",
                    "10_10",
                    "5_5_5",
                ],
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": 10 ** trial.suggest_int("alpha_optuna_exp", -4, 0, step=1),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "adaptive"]
            ),
            # suggest values between 1000 and 10000, but log scale
            "max_iter": trial.suggest_int("max_iter_optuna", 1, 10, log=True) * 1000,
        }
        config_ml["hidden_layers"] = hidden_layers_dict[
            config_ml["hidden_layers_optuna"]
        ]
    elif config_training["model"] == "rf":
        config_ml = {
            "n_estimators": 10
            * trial.suggest_int("n_estimators_optuna", 1, 20, log=True),
            "min_samples_leaf": 3 * trial.suggest_int("min_samples_leaf_optuna", 1, 10),
            "max_features": trial.suggest_int("max_features", 3, 8),
            "max_samples": 0.1 * trial.suggest_int("max_samples_optuna", 2, 10),
        }
    config = merge_dicts_safe(config_general, config_ml, config_posthoc, config_lut)
    return config
