import json
import os
import random
import string
import subprocess
import time
from pickle import dump as pickle_dump
from pickle import load as pickle_load

import ee
import numpy as np
import optuna
import pandas as pd
from geemap import df_to_ee, ml
from loguru import logger
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from config.config import get_config
from gee_pipeline.utils import wait_for_task, wait_for_task_id
from rtm_pipeline_python.classes import (
    helper_apply_posthoc_modifications,
    rtm_simulator,
)
from train_pipeline.utilsLoading import load_validation_data
from train_pipeline.utilsOptuna import log_splits, optuna_init_config
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


def retrain_rf_pipeline_and_upload_gee(
    pipeline: Pipeline,
    X_train,
    y_train,
    feature_names,
    target,
    trait,
    study_name,
    trial_config,
) -> None:
    # recreate pandas dataframe with preprocessed X and transformed y
    X_train_transformed = pipeline.named_steps["preprocessor"].transform(X_train)
    y_train_transformed = pipeline.named_steps["regressor"].transformer_.transform(
        y_train
    )  # use transformer_ instead of transformer, because transformer_ is the fitted transformer
    df_X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    df_y_train_transformed = pd.DataFrame(y_train_transformed, columns=target)
    df_train_transformed = pd.concat(
        [df_X_train_transformed, df_y_train_transformed], axis=1
    )
    # save to csv
    local_lut_folder = os.path.join("data", "train_pipeline", "output", "models", trait)
    local_lut_transformed_filename = f"lut_transformed_{study_name}.csv"
    df_train_transformed.to_csv(
        os.path.join(local_lut_folder, local_lut_transformed_filename),
        index=False,
    )

    gcs_folder_name = CONFIG_GEE_PIPELINE["GCLOUD_FOLDERS"]["TEMP_FOLDER"]
    # upload to google cloud storage
    random_string = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )  # avoid unnecessary filename collisions
    filename_gcs = os.path.join(gcs_folder_name, f"{random_string}.csv")

    # Execute the gsutil command
    subprocess.run(
        f"gsutil cp {os.path.join(local_lut_folder, local_lut_transformed_filename)} {filename_gcs}",
        shell=True,
        check=True,
    )

    # time.sleep(10)
    # import to earth engine using earthengine upload table/
    asset_id = f"{CONFIG_GEE_PIPELINE['GEE_FOLDERS']['MODEL_RF_LUT']}/{local_lut_transformed_filename.removesuffix('.csv')}"
    output = subprocess.run(
        f"{CONFIG_GEE_PIPELINE['CONDA_PATH']}/bin/earthengine upload table --asset_id={asset_id} {filename_gcs}",
        shell=True,
        check=True,
        capture_output=True,
    )
    # extract task id from output
    task_id = output.stdout.decode("utf-8").split("ID: ")[1].strip()

    wait_for_task_id(task_id)

    fc_train_transformed = ee.FeatureCollection(asset_id).select(
        [*feature_names, trait]
    )

    rf_gee_params = {
        "numberOfTrees": trial_config["n_estimators"],
        "variablesPerSplit": trial_config["max_features"],
        "minLeafPopulation": trial_config["min_samples_leaf"],
        "bagFraction": trial_config["max_samples"],
        "seed": 42,
    }
    # train rf model on resampled data
    ee_rf_model = (
        ee.Classifier.smileRandomForest(**rf_gee_params)
        .setOutputMode("REGRESSION")
        .train(
            features=fc_train_transformed,
            classProperty=trait,
            inputProperties=feature_names,
        )
    )

    # save random forest model directly to earth engine asset for later use during prediction
    model_filename = f"model_{study_name}"
    ee_rf_model_filename = (
        f"{CONFIG_GEE_PIPELINE['GEE_FOLDERS']['MODEL_RF_LUT']}/{model_filename}"
    )

    classifier_export_task = ee.batch.Export.classifier.toAsset(
        classifier=ee_rf_model,
        assetId=ee_rf_model_filename,
        description=f"Exporting classifier: {model_filename}",
    )
    classifier_export_task.start()
    # wait for task to finish
    logger.debug("Waiting for classifier export task to finish...")
    wait_for_task(classifier_export_task)


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

    df = lut_simulator.generate_lut()
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

    if config["posthoc_modifications"]:
        df = helper_apply_posthoc_modifications(df, trait, config)

    ### load and prepare validation data
    df_val_trait = load_validation_data(return_site=True)[trait]

    df_val_trait = df_val_trait[
        df_val_trait["ECO_ID"].isin(config["list_ecoids_in_validation"][trait])
    ]

    df_val_trait_dict = {
        eco: df_val_trait[df_val_trait["ECO_ID"] == eco].drop(columns="ECO_ID")
        for eco in config["list_ecoids_in_validation"][trait]
    }

    X, y = df[feature_names], df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

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
    }, {eco: df[target] for eco, df in df_val_train_current.items()}
    X_val_test, y_val_test = {
        eco: df.drop(columns=[*target, "site"])
        for eco, df in df_val_test_current.items()
    }, {eco: df[target] for eco, df in df_val_test_current.items()}

    # instantiate new model instance for each fold
    model = get_model(config)
    pipeline = get_pipeline(model, config)

    pipeline.fit(X_train, y_train)
    # also predict on the simualted training data
    y_sim_train_pred = pipeline.predict(X_train)
    y_sim_test_pred = pipeline.predict(X_val)

    y_val_train_pred = pipeline.predict(
        pd.concat([X_val_train[eco] for eco in sorted(X_val_train.keys())])
    )
    y_val_test_pred = pipeline.predict(
        pd.concat([X_val_test[eco] for eco in sorted(X_val_test.keys())])
    )

    y_val_train = np.concatenate(
        [y_val_train[eco].values.reshape(-1) for eco in sorted(X_val_train.keys())]
    )
    y_val_test = np.concatenate(
        [y_val_test[eco].values.reshape(-1) for eco in sorted(X_val_test.keys())]
    )

    if config["model"] == "rf" and save_model:
        logger.debug(
            f"GEE: Retraining Random Forest model for trait {trait} with trial number {trial.number} with similar hyperparams using ee.Classifier.smileRandomForest"
        )

        retrain_rf_pipeline_and_upload_gee(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            target=target,
            trait=trait,
            study_name=trial.user_attrs["config"]["optuna_study_name"],
            trial_config=config,
        )

    # limit prediction range
    y_val_train_pred = limit_prediction_range(y_val_train_pred, trait)

    scores_sim_train_rmse = root_mean_squared_error(y_train, y_sim_train_pred)
    scores_sim_test_rmse = root_mean_squared_error(y_val, y_sim_test_pred)
    scores_sim_train_mae = mean_absolute_error(y_train, y_sim_train_pred)
    scores_sim_test_mae = mean_absolute_error(y_val, y_sim_test_pred)
    scores_sim_train_r2 = r2_score(y_train, y_sim_train_pred)
    scores_sim_test_r2 = r2_score(y_val, y_sim_test_pred)

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
            pickle_dump(pipeline, f)
        with open(os.path.join(save_folder, f"{model_filename}.json"), "w") as f:
            # also write eval metrics to json
            json.dump(eval_metrics, f)

        with open(full_model_path, "rb") as f:
            pipeline_pickle = pickle_load(f)

        with open(os.path.join(save_folder, f"{model_filename}_config.json"), "w") as f:
            json.dump(trial.user_attrs["config"], f)

        # assert that predictions are the same / close
        assert np.allclose(
            pipeline.predict(X_val_test[list(X_val_test.keys())[0]]),
            pipeline_pickle.predict(X_val_test[list(X_val_test.keys())[0]]),
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
            seed=None, n_startup_trials=500, constant_liar=True, multivariate=True
        )
        study = optuna.create_study(
            storage=storage, study_name=study_name, sampler=sampler
        )
        logger.info(f"Study '{study_name}' created.")
    else:
        sampler = TPESampler(
            seed=None, n_startup_trials=500, constant_liar=True, multivariate=True
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
