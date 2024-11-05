import json
import os
import random
import string
import subprocess
import tempfile
import time
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from typing import Optional

import ee
import numpy as np
import optuna
import pandas as pd
from geemap import df_to_ee, ml
from loguru import logger
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, HalvingGridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config.config import get_config
from gee_pipeline.utils import wait_for_task, wait_for_task_id
from rtm_pipeline_python.classes import (
    helper_apply_posthoc_modifications,
    rtm_simulator,
)
from train_pipeline.utilsLoading import load_validation_data
from train_pipeline.utilsOptuna import log_splits, optuna_init_config
from train_pipeline.utilsPlotting import plot_predicted_vs_true
from train_pipeline.utilsTraining import (
    get_model,
    get_pipeline,
    limit_prediction_range,
    merge_dicts_safe,
    r2_score_oos,
)

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


def save_lut_and_ranges(df, trait, save_folder, study_name, bands) -> None:
    df.to_csv(
        os.path.join(
            save_folder,
            f"lut_{study_name}.csv",
        ),
        index=False,
    )

    # save min and max of band values for later masking:
    min_max_band_values = {
        band: {"min": min(df[band]), "max": max(df[band])} for band in bands
    }
    # save min and max of trait values for later masking:
    min_max_label_values = {trait: {"min": min(df[trait]), "max": max(df[trait])}}

    with open(
        os.path.join(
            save_folder,
            f"min_max_band_values_{study_name}.json",
        ),
        "w",
    ) as f:
        json.dump(min_max_band_values, f)

    with open(
        os.path.join(
            save_folder,
            f"min_max_label_values_{study_name}.json",
        ),
        "w",
    ) as f:
        json.dump(min_max_label_values, f)

    return None


def transform_X_y(
    X: pd.DataFrame,
    y: pd.DataFrame,
    pipeline: Pipeline,
    feature_names: list,
    target: str,
) -> pd.DataFrame:
    X_transformed = pipeline.named_steps["preprocessor"].transform(X[feature_names])
    y_transformed = pipeline.named_steps["regressor"].transformer_.transform(y[target])
    df_X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    df_y_transformed = pd.DataFrame(y_transformed, columns=target)

    if "uuid" in X.columns:
        # add uuid to transformed data
        df_X_transformed["uuid"] = X["uuid"].values
        df_y_transformed["uuid"] = y["uuid"].values
        df_transformed = pd.merge(df_X_transformed, df_y_transformed, on="uuid")
    else:
        df_transformed = pd.concat([df_X_transformed, df_y_transformed], axis=1)
    return df_transformed


def upload_to_ee_via_gcs(df: pd.DataFrame, asset_name: str) -> str:
    gcs_folder_name = CONFIG_GEE_PIPELINE["GCLOUD_FOLDERS"]["TEMP_FOLDER"]
    random_string = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )  # avoid unnecessary filename collisions
    filename_gcs = os.path.join(gcs_folder_name, f"{random_string}.csv")

    # save df to temp dir locally
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp:
        df.to_csv(temp.name, index=False)
        subprocess.run(
            f"gsutil cp {temp.name} {filename_gcs}",
            shell=True,
            check=True,
        )

    asset_id = f"{CONFIG_GEE_PIPELINE['GEE_FOLDERS']['MODEL_RF_LUT']}/{asset_name.removesuffix('.csv')}"
    output = subprocess.run(
        f"{CONFIG_GEE_PIPELINE['CONDA_PATH']}/bin/earthengine upload table --asset_id={asset_id} {filename_gcs}",
        shell=True,
        check=True,
        capture_output=True,
    )

    # extract task id from output
    task_id = output.stdout.decode("utf-8").split("ID: ")[1].strip()

    wait_for_task_id(task_id)

    return asset_id


def objective(trial, save_model=False):

    # try:

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

    ####
    # Generate LUT (simulated training data)
    ####
    logger.info(f"Generating LUT for trait {trait}")

    df = lut_simulator.generate_lut()

    # report min and max of trait values
    trial.set_user_attr(f"min_simulated_{trait}", min(df[trait]))
    trial.set_user_attr(f"max_simulated_{trait}", max(df[trait]))

    df = helper_apply_posthoc_modifications(df, trait, config)

    if save_model:
        save_folder = os.path.join(
            "data",
            "train_pipeline",
            "output",
            "models",
            trait,
        )
        study_name = trial.user_attrs["config"]["optuna_study_name"]
        save_lut_and_ranges(
            df=df,
            trait=trait,
            save_folder=save_folder,
            study_name=study_name,
            bands=bands,
        )

    X, y = df[feature_names], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    #####
    # Load validation data: and split into train and test
    #####

    ### load and prepare validation data
    df_val_trait = load_validation_data(return_site=True)[trait]

    df_val_trait = df_val_trait[
        df_val_trait["ECO_ID"].isin(config["list_ecoids_in_validation"][trait])
    ]

    df_val_trait_dict = {
        eco: df_val_trait[df_val_trait["ECO_ID"] == eco].drop(columns="ECO_ID")
        for eco in config["list_ecoids_in_validation"][trait]
    }

    # find groupings for GroupKFold
    skf = GroupKFold(n_splits=config["group_k_fold_splits"])
    splits = list(skf.split(df_val_trait, groups=df_val_trait["ECO_ID"]))

    val_eco_train_split_indices, val_eco_test_split_indices = splits[
        config["group_k_fold_current_split"]
    ]

    log_splits(splits, df_val_trait, current_fold=config["group_k_fold_current_split"])
    # convert from indices to group values
    val_ecos_train = list(
        set(df_val_trait["ECO_ID"].values[val_eco_train_split_indices])
    )
    val_ecos_test = list(set(df_val_trait["ECO_ID"].values[val_eco_test_split_indices]))

    df_val_train_current = {eco: df_val_trait_dict[eco] for eco in val_ecos_train}
    df_val_test_current = {eco: df_val_trait_dict[eco] for eco in val_ecos_test}

    X_val_train, y_val_train = {
        eco: df.drop(columns=[*target, "site"])
        for eco, df in df_val_train_current.items()
    }, {eco: df[[*target, "uuid"]] for eco, df in df_val_train_current.items()}
    X_val_test, y_val_test = {
        eco: df.drop(columns=[*target, "site"])
        for eco, df in df_val_test_current.items()
    }, {eco: df[[*target, "uuid"]] for eco, df in df_val_test_current.items()}

    X_val_train = pd.concat(
        [X_val_train[eco] for eco in sorted(df_val_train_current.keys())]
    )
    X_val_test = pd.concat(
        [X_val_test[eco] for eco in sorted(df_val_test_current.keys())]
    )
    y_val_train = pd.concat(
        [y_val_train[eco] for eco in sorted(df_val_train_current.keys())]
    )
    y_val_test = pd.concat(
        [y_val_test[eco] for eco in sorted(df_val_test_current.keys())]
    )

    ####
    # Train Model
    ####

    logger.info(f"Training model for trait {trait} using HalvingGridSearchCV")

    # instantiate new model instance for each fold
    param_grid = config["mlp_grid"].copy()

    hidden_layers_dict = {
        "5": (5,),
        "10": (10,),
        "5_5": (5, 5),
        "10_5": (10, 5),
    }

    # replace hidden_layer_sizes with tuple
    param_grid["hidden_layer_sizes"] = [
        hidden_layers_dict[hidden_layer]
        for hidden_layer in param_grid["hidden_layer_sizes"]
    ]

    # prepend 'regressor__regressor__' to all keys in param_grid
    param_grid = {
        f"regressor__regressor__{key}": value for key, value in param_grid.items()
    }

    # from sklearn.neural_network import MLPRegressor
    model = get_model(config)
    pipeline = get_pipeline(model, config)

    halving_search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        random_state=42,
        n_jobs=6,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        factor=5,
    )

    halving_search.fit(X_train, y_train)

    best_pipeline = halving_search.best_estimator_

    # refit with all simualted training data for best pipeline
    best_pipeline.fit(X_train, y_train)

    # also predict on the simualted training data
    y_sim_train_pred = best_pipeline.predict(X_train)
    y_sim_test_pred = best_pipeline.predict(X_test)

    y_val_train_pred = best_pipeline.predict(X_val_train)
    y_val_test_pred = best_pipeline.predict(X_val_test)
    if "uuid" in X_val_train.columns:
        assert "uuid" in X_val_test.columns
        y_val_train_pred = pd.DataFrame(y_val_train_pred, columns=target)
        y_val_train_pred["uuid"] = X_val_train["uuid"].values
        y_val_test_pred = pd.DataFrame(y_val_test_pred, columns=target)
        y_val_test_pred["uuid"] = X_val_test["uuid"].values

    if save_model:
        # save original validation data / merge X and y
        df_val_train = pd.merge(X_val_train, y_val_train, on="uuid", how="inner")
        df_val_test = pd.merge(X_val_test, y_val_test, on="uuid", how="inner")

        df_val_train.to_csv(
            os.path.join(save_folder, f"df_val_train_{trait}_{study_name}.csv"),
            index=False,
        )
        df_val_test.to_csv(
            os.path.join(save_folder, f"df_val_test_{trait}_{study_name}.csv"),
            index=False,
        )

        # save transformed validation data
        transform_X_y(
            X_val_train, y_val_train, best_pipeline, feature_names, target
        ).to_csv(
            os.path.join(
                save_folder, f"df_val_train_transformed_{trait}_{study_name}.csv"
            ),
            index=False,
        )

        transform_X_y(
            X_val_test, y_val_test, best_pipeline, feature_names, target
        ).to_csv(
            os.path.join(
                save_folder, f"df_val_test_transformed_{trait}_{study_name}.csv"
            ),
            index=False,
        )

    # limit prediction range
    y_val_train_pred = limit_prediction_range(y_val_train_pred, trait)

    ####
    # Evaluate Model
    ####

    scores_sim_train_rmse = root_mean_squared_error(y_train, y_sim_train_pred)
    scores_sim_test_rmse = root_mean_squared_error(y_test, y_sim_test_pred)
    scores_sim_train_mae = mean_absolute_error(y_train, y_sim_train_pred)
    scores_sim_test_mae = mean_absolute_error(y_test, y_sim_test_pred)
    scores_sim_train_r2 = r2_score(y_train, y_sim_train_pred)
    scores_sim_test_r2 = r2_score(y_test, y_sim_test_pred)

    if "uuid" in y_val_train_pred.columns and "uuid" in y_val_train.columns:
        assert (y_val_train_pred["uuid"].values == y_val_train["uuid"].values).all()
        score_val_train_rmse = root_mean_squared_error(
            y_val_train[trait], y_val_train_pred[trait]
        )
        score_val_test_rmse = root_mean_squared_error(
            y_val_test[trait], y_val_test_pred[trait]
        )
        score_val_train_mae = mean_absolute_error(
            y_val_train[trait], y_val_train_pred[trait]
        )
        score_val_test_mae = mean_absolute_error(
            y_val_test[trait], y_val_test_pred[trait]
        )
        score_val_train_r2 = r2_score(y_val_train[trait], y_val_train_pred[trait])
        score_val_test_r2 = r2_score(y_val_test[trait], y_val_test_pred[trait])
        score_val_test_r2_oos = r2_score_oos(
            y_true=y_val_test[trait],
            y_pred=y_val_test_pred[trait],
            y_true_train=y_val_train[trait],
        )
    else:
        score_val_train_rmse = root_mean_squared_error(y_val_train, y_val_train_pred)
        score_val_test_rmse = root_mean_squared_error(y_val_test, y_val_test_pred)
        score_val_train_mae = mean_absolute_error(y_val_train, y_val_train_pred)
        score_val_test_mae = mean_absolute_error(y_val_test, y_val_test_pred)
        score_val_train_r2 = r2_score(y_val_train, y_val_train_pred)
        score_val_test_r2 = r2_score(y_val_test, y_val_test_pred)
        score_val_test_r2_oos = r2_score_oos(
            y_true=y_val_test, y_pred=y_val_test_pred, y_true_train=y_val_train
        )

    if not save_model:
        # log current split and eco_ids in train and test split
        trial.set_user_attr(
            "val_ecos_train", ", ".join([str(eco) for eco in val_ecos_train])
        )
        trial.set_user_attr(
            "val_ecos_test", ", ".join([str(eco) for eco in val_ecos_test])
        )

        # Log simulation values
        trial.set_user_attr(
            "sim_train_rmse",
            min(scores_sim_train_rmse, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr(
            "sim_test_rmse",
            min(scores_sim_test_rmse, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr(
            "sim_train_mae",
            min(scores_sim_train_mae, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr(
            "sim_test_mae",
            min(scores_sim_test_mae, config["optuna_report_thresh"][trait]),
        )
        trial.set_user_attr("sim_train_r2", max(scores_sim_train_r2, -1))
        trial.set_user_attr("sim_test_r2", max(scores_sim_test_r2, -1))

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
    else:
        eval_metrics = {}
        eval_metrics["sim_train_rmse"] = scores_sim_train_rmse
        eval_metrics["sim_test_rmse"] = scores_sim_test_rmse
        eval_metrics["sim_train_mae"] = scores_sim_train_mae
        eval_metrics["sim_test_mae"] = scores_sim_test_mae
        eval_metrics["sim_train_r2"] = scores_sim_train_r2
        eval_metrics["sim_test_r2"] = scores_sim_test_r2
        eval_metrics["val_train_rmse"] = score_val_train_rmse
        eval_metrics["val_test_rmse"] = score_val_test_rmse
        eval_metrics["val_train_mae"] = score_val_train_mae
        eval_metrics["val_test_mae"] = score_val_test_mae
        eval_metrics["val_train_r2"] = score_val_train_r2
        eval_metrics["val_test_r2"] = score_val_test_r2
        eval_metrics["val_test_r2_oos"] = score_val_test_r2_oos

    # save model
    if save_model:

        logger.debug(f"Saving model for trait {trait} with trial number {trial.number}")

        os.makedirs(save_folder, exist_ok=True)
        model_filename = f"model_{trial.user_attrs['config']['optuna_study_name']}"
        full_model_path = os.path.join(save_folder, f"{model_filename}.pkl")
        with open(full_model_path, "wb") as f:
            pickle_dump(best_pipeline, f)
        with open(os.path.join(save_folder, f"{model_filename}.json"), "w") as f:
            # also write eval metrics to json
            json.dump(eval_metrics, f)

        with open(full_model_path, "rb") as f:
            pipeline_pickle = pickle_load(f)

        with open(os.path.join(save_folder, f"{model_filename}_config.json"), "w") as f:
            json.dump(trial.user_attrs["config"], f)

        with open(os.path.join(save_folder, f"{model_filename}_split.json"), "w") as f:
            json.dump(
                {
                    "val_ecos_train": trial.user_attrs["val_ecos_train"],
                    "val_ecos_test": trial.user_attrs["val_ecos_test"],
                },
                f,
            )

        # assert that predictions are the same / close
        assert np.allclose(
            best_pipeline.predict(X_val_test),
            pipeline_pickle.predict(X_val_test),
        )

        ####
        # Plotting
        ####

        # plot predicted vs true for simulated test set
        plot_predicted_vs_true(
            y_true=y_train,
            y_pred=y_sim_train_pred,
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_sim_train_{model_filename}.png"
            ),
            title=f"Predicted vs True for trait {trait} on Simulated Training Set",
        )

        # plot predicted vs true for simulated test set
        plot_predicted_vs_true(
            y_true=y_test,
            y_pred=y_sim_test_pred,
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_sim_test_{model_filename}.png"
            ),
            title=f"Predicted vs True for trait {trait} on Simulated Test Set",
        )

        # plot predicted vs true for validation train and test
        plot_predicted_vs_true(
            y_true=y_val_train[trait],
            y_pred=y_val_train_pred[trait],
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_val_train_{model_filename}.png"
            ),
            title=f"Predicted vs True for trait {trait} on Validation Training Set",
        )

        plot_predicted_vs_true(
            y_true=y_val_test[trait],
            y_pred=y_val_test_pred[trait],
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_val_test_{model_filename}.png"
            ),
            title=f"Predicted vs True for trait {trait} on Validation Test Set",
        )
    return score_val_train_rmse


# except Exception as e:
#     logger.error(f"Error in objective function: {e}, Failed trial and continue.")
#     return float("nan")


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
            seed=None, n_startup_trials=1000, constant_liar=True, multivariate=True
        )
        study = optuna.create_study(
            storage=storage, study_name=study_name, sampler=sampler
        )
        logger.info(f"Study '{study_name}' created.")
    else:
        sampler = TPESampler(
            seed=None, n_startup_trials=1000, constant_liar=True, multivariate=True
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
