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
    r2_score_oos,
    rf_get_size_of_string,
)


def objective(trial):

    try:

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
                for ecoregion in config["list_ecoids_in_validation"][trait]
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
            df_val_trait["ECO_ID"].isin(config["list_ecoids_in_validation"][trait])
        ]

        df_val_trait_dict = {
            eco: df_val_trait[df_val_trait["ECO_ID"] == eco].drop(columns="ECO_ID")
            for eco in config["list_ecoids_in_validation"][trait]
        }

        if config["ecoregion_level"]:
            X = {eco: df_eco[feature_names] for eco, df_eco in df.items()}
            y = {eco: df_eco[target] for eco, df_eco in df.items()}
        else:
            X, y = df[feature_names], df[target]

        # find groupings for GroupKFold
        skf = GroupKFold(n_splits=config["group_k_fold_splits"])
        splits = list(skf.split(df_val_trait, groups=df_val_trait["ECO_ID"]))

        val_eco_train_split_indices, val_eco_test_split_indices = splits[
            config["group_k_fold_current_split"]
        ]

        log_splits(
            splits, df_val_trait, current_fold=config["group_k_fold_current_split"]
        )
        # convert from indices to group values
        val_ecos_train = list(
            set(df_val_trait["ECO_ID"].values[val_eco_train_split_indices])
        )
        val_ecos_test = list(
            set(df_val_trait["ECO_ID"].values[val_eco_test_split_indices])
        )

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
            pipeline.fit(X, y, ecoregions=config["list_ecoids_in_validation"][trait])
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
                    y_val_train[eco].values.reshape(-1)
                    for eco in sorted(y_val_train.keys())
                ]
            )
            y_val_test = np.concatenate(
                [
                    y_val_test[eco].values.reshape(-1)
                    for eco in sorted(y_val_test.keys())
                ]
            )
        else:
            pipeline.fit(X, y)
            y_val_train_pred = pipeline.predict(
                pd.concat([X_val_train[eco] for eco in sorted(X_val_train.keys())])
            )
            y_val_test_pred = pipeline.predict(
                pd.concat([X_val_test[eco] for eco in sorted(X_val_test.keys())])
            )

            y_val_train = np.concatenate(
                [
                    y_val_train[eco].values.reshape(-1)
                    for eco in sorted(X_val_train.keys())
                ]
            )
            y_val_test = np.concatenate(
                [
                    y_val_test[eco].values.reshape(-1)
                    for eco in sorted(X_val_test.keys())
                ]
            )

        # save length of strings for model rf:
        if config["model"] == "rf":
            if config["ecoregion_level"]:
                rf_model = (
                    pipeline.per_ecoregion_pipeline_[
                        list(config["list_ecoids_in_validation"][trait])[0]
                    ]
                    .named_steps["regressor"]
                    .regressor_
                )
            else:
                rf_model = pipeline.named_steps["regressor"].regressor_
            string_size = rf_get_size_of_string(rf_model, feature_names=feature_names)
            trial.set_user_attr("string_size_mb", string_size["megabytes"])

            # if string size is too large, give warning / since earth engine has limit of 10MB / and we want to run at least 6 models in the same export. 9 MB + 1 MB Overhead
            if string_size["megabytes"] > 1.5:
                logger.warning(f"String size is larger than 1.5 MB: {string_size}")

        # limit prediction range
        y_val_train_pred = limit_prediction_range(y_val_train_pred, trait)

        score_val_train_rmse = root_mean_squared_error(y_val_train, y_val_train_pred)
        score_val_test_rmse = root_mean_squared_error(y_val_test, y_val_test_pred)
        score_val_train_mae = mean_absolute_error(y_val_train, y_val_train_pred)
        score_val_test_mae = mean_absolute_error(y_val_test, y_val_test_pred)
        score_val_train_r2 = r2_score(y_val_train, y_val_train_pred)
        score_val_test_r2 = r2_score(y_val_test, y_val_test_pred)
        score_val_test_r2_oos = r2_score_oos(
            y_true=y_val_test, y_pred=y_val_test_pred, y_true_train=y_val_train
        )

        # log current split and eco_ids in train and test split
        trial.set_user_attr(
            "val_ecos_train", ", ".join([str(eco) for eco in val_ecos_train])
        )
        trial.set_user_attr(
            "val_ecos_test", ", ".join([str(eco) for eco in val_ecos_test])
        )

        # Log additional values / but set max or min values to avoid errors
        trial.set_user_attr(
            "val_train_rmse",
            min(score_val_train_rmse, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr(
            "val_test_rmse",
            min(score_val_test_rmse, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr(
            "val_train_mae",
            min(score_val_train_mae, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr(
            "val_test_mae",
            min(score_val_test_mae, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr("val_train_r2", max(score_val_train_r2, -1))
        trial.set_user_attr("val_test_r2", max(score_val_test_r2, -1))
        trial.set_user_attr("val_test_r2_oos", max(score_val_test_r2_oos, -1))
        trial.set_user_attr("config", config)

        return score_val_train_rmse

    except Exception as e:
        logger.error(f"Error in objective function: {e}, Failed trial and continue.")
        return float("nan")


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
        sampler = TPESampler(
            seed=None, n_startup_trials=100, constant_liar=True, multivariate=True
        )
        study = optuna.create_study(
            storage=storage, study_name=study_name, sampler=sampler
        )
        logger.info(f"Study '{study_name}' created.")
    else:
        sampler = TPESampler(
            seed=None, n_startup_trials=100, constant_liar=True, multivariate=True
        )
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=sampler
        )
        logger.info(f"Study '{study_name}' loaded.")

    study.optimize(objective, n_trials=2000)

    # get the best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    main()
