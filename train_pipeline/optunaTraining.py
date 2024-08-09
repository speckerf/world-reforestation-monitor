import os

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

from config.config import get_config
from rtm_pipeline_python.classes import (
    helper_apply_posthoc_modifications,
    rtm_simulator,
)
from train_pipeline.utilsLoading import load_validation_data
from train_pipeline.utilsOptuna import log_splits, optuna_init_config
from train_pipeline.utilsTraining import (
    EcoregionSpecificModel,
    get_model,
    get_pipeline,
    limit_prediction_range,
    merge_dicts_safe,
)


def objective(trial):

    config_optuna = optuna_init_config(trial)
    config_general = get_config("train_pipeline")
    config = merge_dicts_safe(config_general, config_optuna)

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    trait = config["trait"]
    if config["use_angles_for_prediction"]:
        angles = ["tts", "tto", "psi"]
    else:
        angles = []
    feature_names = [*bands, *angles]
    target = [trait]

    lut_simulator = rtm_simulator(
        config,
        r_script_path=os.path.join(
            "rtm_pipeline_R",
            "src",
            "run_prosail.R",
        ),
    )

    if config["ecoregion_level"]:
        df = {
            ecoregion: lut_simulator.generate_lut(ecoregion)
            for ecoregion in config["list_ecoids_in_lai_validation"]
        }
        if config["posthoc_modifications"]:
            df = {
                eco: helper_apply_posthoc_modifications(single_df, trait, config)
                for eco, single_df in df.items()
            }
    else:
        df = lut_simulator.generate_lut()
        if config["posthoc_modifications"]:
            df = helper_apply_posthoc_modifications(df, trait, config)

    ### load and prepare validation data
    df_val_trait = load_validation_data(return_site=True)[trait]

    # df_val_trait = df_val_trait[df_val_trait[trait] > 0.001]
    # df_val_trait["site"] = df_val_trait["site"].apply(lambda x: x.split("_")[0])
    df_val_trait = df_val_trait[
        df_val_trait["ECO_ID"].isin(config["list_ecoids_in_lai_validation"])
    ]

    df_val_trait_dict = {
        eco: df_val_trait[df_val_trait["ECO_ID"] == eco].drop(columns="ECO_ID")
        for eco in config["list_ecoids_in_lai_validation"]
    }

    if config["ecoregion_level"]:
        X = {eco: df_eco[feature_names] for eco, df_eco in df.items()}
        y = {eco: df_eco[target] for eco, df_eco in df.items()}
    else:
        X, y = df[feature_names], df[target]

    # find groupings for GroupKFold
    skf = GroupKFold(n_splits=3)
    splits = list(skf.split(df_val_trait, groups=df_val_trait["ECO_ID"]))
    log_splits(splits, df_val_trait)
    # convert from indices to group values
    val_eco_train_splits = [
        list(set(df_val_trait["ECO_ID"].values[split[0]])) for split in splits
    ]
    val_eco_test_splits = [
        list(set(df_val_trait["ECO_ID"].values[split[1]])) for split in splits
    ]

    scores_val_train_rmse = []
    scores_val_test_rmse = []
    scores_val_train_mae = []
    scores_val_test_mae = []
    scores_val_train_r2 = []
    scores_val_test_r2 = []

    for val_ecos_train, val_ecos_test in zip(val_eco_train_splits, val_eco_test_splits):
        df_val_train_current = {eco: df_val_trait_dict[eco] for eco in val_ecos_train}
        df_val_test_current = {eco: df_val_trait_dict[eco] for eco in val_ecos_test}

        X_val_train, y_val_train = {
            eco: df.drop(columns=[*target, "site"])
            for eco, df in df_val_train_current.items()
        }, {eco: df[target] for eco, df in df_val_train_current.items()}
        X_val_test, y_val_test = {
            eco: df.drop(columns=[*target, "site"])
            for eco, df in df_val_test_current.items()
        }, {eco: df[target] for eco, df in df_val_test_current.items()}

        # instantiate new model instance for each fold
        model = get_model(config)
        pipeline = get_pipeline(model, config)
        if config["ecoregion_level"]:
            pipeline = EcoregionSpecificModel(pipeline, config)
            pipeline.fit(X, y, ecoregions=config["list_ecoids_in_lai_validation"])
            y_val_train_pred = pipeline.predict(X_val_train)
            y_val_test_pred = pipeline.predict(X_val_test)

            # sort dictionary by key
            y_val_train_pred = np.concatenate(
                [y_val_train_pred[eco] for eco in sorted(y_val_train_pred.keys())]
            )
            y_val_test_pred = np.concatenate(
                [y_val_test_pred[eco] for eco in sorted(y_val_test_pred.keys())]
            )

            # sort true values by key
            y_val_train = np.concatenate(
                [
                    y_val_train[eco].values.squeeze()
                    for eco in sorted(y_val_train.keys())
                ]
            )
            y_val_test = np.concatenate(
                [y_val_test[eco].values.squeeze() for eco in sorted(y_val_test.keys())]
            )
        else:
            pipeline = pipeline
            pipeline.fit(X, y)
            y_val_train_pred = pipeline.predict(
                pd.concat([X_val_train[eco] for eco in sorted(X_val_train.keys())])
            )
            y_val_test_pred = pipeline.predict(
                pd.concat([X_val_test[eco] for eco in sorted(X_val_test.keys())])
            )

            y_val_train = np.concatenate(
                [
                    y_val_train[eco].values.squeeze()
                    for eco in sorted(X_val_train.keys())
                ]
            )
            y_val_test = np.concatenate(
                [y_val_test[eco].values.squeeze() for eco in sorted(X_val_test.keys())]
            )

        # limit prediction range
        y_val_train_pred = limit_prediction_range(y_val_train_pred, trait)

        scores_val_train_rmse.append(
            root_mean_squared_error(y_val_train, y_val_train_pred)
        )

        scores_val_test_rmse.append(
            root_mean_squared_error(y_val_test, y_val_test_pred)
        )
        scores_val_train_mae.append(mean_absolute_error(y_val_train, y_val_train_pred))
        scores_val_test_mae.append(mean_absolute_error(y_val_test, y_val_test_pred))
        scores_val_train_r2.append(r2_score(y_val_train, y_val_train_pred))
        scores_val_test_r2.append(r2_score(y_val_test, y_val_test_pred))

    # Log additional values / but set max or min values to avoid errors
    trial.set_user_attr("val_train_rmse", np.min(np.mean(scores_val_train_rmse), 5))
    trial.set_user_attr("val_test_rmse", np.min(np.mean(scores_val_test_rmse), 5))
    trial.set_user_attr("val_train_mae", np.min(np.mean(scores_val_train_mae), 5))
    trial.set_user_attr("val_test_mae", np.min(np.mean(scores_val_test_mae), 5))
    trial.set_user_attr("val_train_r2", np.max(np.mean(scores_val_train_r2), -1))
    trial.set_user_attr("val_test_r2", np.max(np.mean(scores_val_test_r2), -1))
    trial.set_user_attr("config", config)

    return np.mean(scores_val_train_rmse)


def main():

    config_general = get_config("train_pipeline")

    logger.info(f"Start Optuna training for trait {config_general['trait']}")
    logger.info(f"Optuna study name: {config_general['optuna_study_name']}")
    logger.info(f"Optuna storage: {config_general['optuna_storage']}")

    # connect to storage
    storage = RDBStorage(
        url=config_general["optuna_storage"],
    )
    study_name = config_general["optuna_study_name"]

    # check if study already exists
    study_exists = study_name in [
        s.study_name for s in optuna.study.get_all_study_summaries(storage=storage)
    ]

    # Create the study if it doesn't exist
    if not study_exists:
        sampler = TPESampler(seed=10, n_startup_trials=100)
        study = optuna.create_study(
            storage=storage, study_name=study_name, sampler=sampler
        )
        logger.info(f"Study '{study_name}' created.")
    else:
        sampler = TPESampler(n_startup_trials=100)
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=sampler
        )
        logger.info(f"Study '{study_name}' loaded.")

    if not study_exists:
        study.enqueue_trial(
            params={
                "model": "rf",
                "ecoregion_level": False,
                "use_angles_for_prediction": True,
                "posthoc_modifications": False,
                "nirv_norm": True,
                "modify_rsoil": False,
                "add_noise": True,
                "noise_type": "atbd",
                "num_spectra_optuna": 18,
                "parameter_setup": "foliar_codistribution",
                "n_estimators_optuna": 5,
                "max_depth_optuna": 3,
                "min_samples_split_optuna": 5,
                "min_samples_leaf_optuna": 5,
                "max_features": 8,
            }
        )
        study.enqueue_trial(
            params={
                "model": "rf",
                "ecoregion_level": False,
                "use_angles_for_prediction": True,
                "posthoc_modifications": True,
                "use_baresoil_insitu": False,
                "use_urban_s2": True,
                "use_water_s2": True,
                "use_snowice_s2": False,
                "use_baresoil_emit": True,
                "use_baresoil_s2": False,
                "n_baresoil_insitu_optuna": 15,
                "n_baresoil_emit_optuna": 4,
                "n_urban_s2_optuna": 1,
                "nirv_norm": True,
                "modify_rsoil": False,
                "add_noise": True,
                "noise_type": "atbd",
                "num_spectra_optuna": 18,
                "parameter_setup": "foliar_codistribution",
                "n_estimators_optuna": 5,
                "max_depth_optuna": 3,
                "min_samples_split_optuna": 5,
                "min_samples_leaf_optuna": 5,
                "max_features": 8,
            }
        )
        study.enqueue_trial(
            params={
                "model": "mlp",
                "ecoregion_level": False,
                "use_angles_for_prediction": True,
                "posthoc_modifications": True,
                "use_baresoil_insitu": False,
                "use_urban_s2": True,
                "use_water_s2": True,
                "use_snowice_s2": False,
                "use_baresoil_emit": True,
                "use_baresoil_s2": False,
                "n_baresoil_insitu_optuna": 15,
                "n_baresoil_emit_optuna": 4,
                "n_urban_s2_optuna": 1,
                "nirv_norm": False,
                "modify_rsoil": False,
                "add_noise": True,
                "noise_type": "atbd",
                "num_spectra_optuna": 18,
                "parameter_setup": "foliar_codistribution",
                "hidden_layers_optuna": "10_10",
                "activation": "tanh",
                "alpha_optuna_exp": -4,
                "learning_rate": "constant",
                "max_iter_optuna": 10,
            }
        )
        study.enqueue_trial(
            params={
                "model": "mlp",
                "ecoregion_level": False,
                "use_angles_for_prediction": True,
                "posthoc_modifications": False,
                "nirv_norm": False,
                "modify_rsoil": False,
                "add_noise": True,
                "noise_type": "atbd",
                "num_spectra_optuna": 18,
                "parameter_setup": "foliar_codistribution",
                "hidden_layers_optuna": "10_10",
                "activation": "tanh",
                "alpha_optuna_exp": -4,
                "learning_rate": "constant",
                "max_iter_optuna": 10,
            }
        )
    study.optimize(objective, n_trials=2000)

    # get the best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    main()
