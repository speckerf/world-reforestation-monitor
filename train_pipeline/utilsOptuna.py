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
                "transform_target", ["logit", "log1p", "None"]
            ),
        }
    elif trait == "fcover":
        config_transform_target = {
            "transform_target": trial.suggest_categorical(
                "transform_target", ["logit", "log1p", "None"]
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
        **config_transform_target,
        "nirv_norm": trial.suggest_categorical("nirv_norm", [True, False]),
        # "nirv_norm": True,
    }

    config_lut = {
        # "add_noise": True,
        "num_spectra": 2500
        * trial.suggest_int("num_spectra_optuna", 1, 20, log=True),
    }

    if trait == "lai":
        config_lut["parameter_setup"] = trial.suggest_categorical(
            "parameter_setup",
            [
                "estevez_2022",
                "foliar_codistribution",
                "kovacs_2023",
                "snap_atbd",
                "wan_2024_lai",
            ],
        )
    # use modified versions for fapar and fcover to let the model learn the range of LAI itsef (because this is decisive for the upper bound of simulated fapar/fcover)
    elif trait == "fapar":
        parameter_setup = trial.suggest_categorical(
            "parameter_setup",
            [
                "estevez_2022_mod",
                "foliar_codistribution_mod",
                "kovacs_2023_mod",
                "snap_atbd_mod",
                "wan_2024_lai_mod",
            ],
        )
        config_lut["parameter_setup"] = parameter_setup
        if parameter_setup != "snap_atbd":
            config_lut["lai_min"] = 0.0
            config_lut["lai_max"] = 15.0
            config_lut["lai_mean"] = trial.suggest_float("lai_mean", 0.0, 5.0, step=0.2)
            config_lut["lai_std"] = trial.suggest_float("lai_std", 0.2, 5.0, step=0.2)

    elif trait == "fcover":
        parameter_setup = trial.suggest_categorical(
            "parameter_setup",
            [
                "estevez_2022_mod",
                "foliar_codistribution_mod",
                "kovacs_2023_mod",
                "snap_atbd_mod",
                "wan_2024_lai_mod",
            ],
        )
        config_lut["parameter_setup"] = parameter_setup
        if parameter_setup != "snap_atbd":
            config_lut["lai_min"] = 0.0
            config_lut["lai_max"] = 15.0
            config_lut["lai_mean"] = trial.suggest_float("lai_mean", 0.0, 5.0, step=0.2)
            config_lut["lai_std"] = trial.suggest_float("lai_std", 0.0, 5.0, step=0.2)

    # other foliar traits
    elif trait == "CHL":
        config_lut["parameter_setup"] = trial.suggest_categorical(
            "parameter_setup",
            [
                "estevez_2022",
                "foliar_codistribution",
                "kovacs_2023",
                "snap_atbd",
                "wan_2024_chl",
            ],
        )
    elif trait == "EWT":
        config_lut["parameter_setup"] = trial.suggest_categorical(
            "parameter_setup",
            [
                "foliar_codistribution",
                "kovacs_2023",
                "snap_atbd",
                "wan_2024_lai",
                "custom_ewt",
            ],
        )
    elif trait == "LMA":
        config_lut["parameter_setup"] = trial.suggest_categorical(
            "parameter_setup",
            [
                "foliar_codistribution",
                "kovacs_2023",
                "snap_atbd",
                "wan_2024_lai",
                "custom_lma",
            ],
        )

    config_lut["additive_noise"] = (
        0.005 * trial.suggest_int("additive_noise_optuna", 1, 7) - 0.005
    )
    config_lut["multiplicative_noise"] = (
        0.01 * trial.suggest_int("multiplicative_noise_optuna)", 1, 11) - 0.01
    )

    config_lut["rsoil_emit_insitu"] = trial.suggest_categorical(
        "rsoil_emit_insitu", ["emit", "insitu"]
    )
    config_lut["rsoil_fraction"] = 0.1 * trial.suggest_int(
        "rsoil_fraction_optuna", 1, 10, log=True
    )

    config_posthoc = {}
    config_posthoc["p_baresoil_insitu"] = (
        0.01 * trial.suggest_int("p_baresoil_insitu_optuna", 1, 11, log=True) - 0.01
    )
    config_posthoc["p_baresoil_s2"] = (
        0.01 * trial.suggest_int("p_baresoil_s2_optuna", 1, 11, log=True) - 0.01
    )
    config_posthoc["p_baresoil_emit"] = (
        0.01 * trial.suggest_int("p_baresoil_emit_optuna", 1, 11, log=True) - 0.01
    )
    config_posthoc["p_urban_s2"] = (
        0.005 * trial.suggest_int("p_urban_s2_optuna", 1, 11, log=True) - 0.005
    )
    # config_posthoc["n_water_s2"] = (
    #     20 * trial.suggest_int("n_water_s2_optuna", 1, 26, log=True) - 20
    # )
    config_posthoc["p_snowice_s2"] = (
        0.005 * trial.suggest_int("p_snowice_s2_optuna", 1, 11, log=True) - 0.005
    )
    # config_posthoc["n_water_s2"] = 0

    config = merge_dicts_safe(config_general, config_posthoc, config_lut)
    return config
