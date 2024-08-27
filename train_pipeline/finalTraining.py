import json
import os
import random
from glob import glob
from pickle import load as pickle_load

import ee
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from config.config import get_config
from train_pipeline.optunaTraining import objective
from train_pipeline.utilsLoading import load_validation_data


def rerun_and_save_best_optuna(config: dict, study=None) -> None:

    if study is None:
        # load the study
        study = optuna.load_study(
            study_name=config["optuna_study_name"], storage=config["optuna_storage"]
        )
    else:
        # create a new study
        study = study
    # rerun best model
    # Retrieve the best trial
    best_trial = study.best_trial

    # if model == 'rf':
    #    take the best trial with user attribute 'string_size_mb' lower than 1
    if config["model"] == "rf":
        trials_filtered = [
            t
            for t in study.trials
            if t.user_attrs.get("string_size_mb", float("inf")) < 1
            and t.value is not None
        ]
        best_trial_filtered = min(trials_filtered, key=lambda t: t.value)
        best_trial_number = best_trial_filtered.number
        best_trial_value = best_trial_filtered.value
        best_trial_params = best_trial_filtered.params
        # run the best model
        objective(best_trial_filtered, save_model=True)

    elif config["model"] == "mlp":
        trials_filtered = [
            t
            for t in study.trials
            if t.params["hidden_layers"] in ["5", "5_5", "10", "5_10"]
            and t.value is not None
        ]

        best_trial_filtered = min(trials_filtered, key=lambda t: t.value)
        best_trial_number = best_trial_filtered.number
        best_trial_value = best_trial_filtered.value
        best_trial_params = best_trial_filtered.params
        # run the best model
        objective(best_trial_filtered, save_model=True)


def rerun_and_save_best_optuna_wrapper(trait: str, config: dict):
    models = ["rf", "mlp"]
    # models = ["rf"]
    testsets = [0, 1, 2]
    # models = ["mlp"]
    # testsets = [0]

    for model in models:
        for testset in testsets:
            study_name = f"optuna-debug-{trait}-{model}-testset{testset}"

            config["optuna_study_name"] = study_name
            config["model"] = model
            config["trait"] = trait
            config["group_k_fold_current_split"] = testset

            # check that study exists
            try:
                study = optuna.load_study(
                    study_name=study_name, storage=config["optuna_storage"]
                )
            except:
                logger.error(f"Study {study_name} does not exist")
                continue

            # run and save best model
            rerun_and_save_best_optuna(config, study=study)


# def test_gee_pipeline_predict(trait: str):
#     """
#     Test the eePipelinePredict function:
#     with constant image, see if same predictions are made
#     """
#     models = load_model_ensemble(trait)

#     bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
#     angles = ["tts", "tto", "psi"]
#     features = bands + angles

#     # create dummy image with constant values:
#     random_reflectances = [random.random() for _ in range(len(bands))]
#     random_angles = [random.random() * 10 for _ in range(len(angles))]
#     random_angles[0] = random_angles[0] * 4
#     random_angles[1] = random_angles[1] * 1
#     random_angles[2] = random_angles[2] * 18

#     img = (
#         ee.ImageCollection(
#             [
#                 *[ee.Image(v).rename(n) for n, v in zip(bands, random_reflectances)],
#                 *[ee.Image(v).rename(n) for n, v in zip(angles, random_angles)],
#             ]
#         )
#         .toBands()
#         .rename(features)
#     )

#     # sample point from image
#     point = ee.Geometry.Point([0, 0])
#     img.sample(point).first().getInfo()

#     # loop over all models
#     for model_name, model in models.items():
#         # if model is rf, continue
#         # if model_name.split("-")[-2] == "rf":
#         #     continue
#         # predict the image
#         img_pred = eePipelinePredict(model["pipeline"], img, trait, model["config"])
#         img_pred_gee = img_pred.sample(point).first().getInfo()["properties"][trait]

#         # predict locally
#         X = np.array([*random_reflectances, *random_angles]).reshape(1, -1)
#         # convert to dataframe
#         X = pd.DataFrame(X, columns=features)
#         y_pred_sklearn = model["pipeline"].predict(X)

#         logger.debug(
#             f"Model {model_name} sklearn_pred: {y_pred_sklearn[0][0]}, GEE_pred: {img_pred_gee}"
#         )
#         if not np.isclose(img_pred_gee, y_pred_sklearn[0][0], rtol=1e-5):
#             logger.error(
#                 f"Model {model_name} prediction is not the same: Difference: {img_pred_gee - y_pred_sklearn[0][0]}"
#             )
#             raise ValueError(
#                 f"Model {model_name} prediction is not the same, Please recheck the eePipelinePredict model conversion."
#             )
#         else:
#             logger.info(f"Model {model_name} predict GEE translation was successful")


def evaluate_model_ensemble(trait: str) -> tuple:
    """
    Evaluate the model ensemble for the given trait
    :param trait: str, trait name
    :return: tuple, predictions_ensemble, y_val
    """
    models = load_model_ensemble(trait)
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]
    features = bands + angles

    # load the validation data
    validation_data = load_validation_data()[trait]
    X_val, y_val = validation_data[features], validation_data[trait]

    predictions = {}
    # loop over all models
    for model_name, model in models.items():
        # predict the validation data
        predictions[model_name] = model.predict(X_val)

    # average the predictions across models
    predictions_ensemble = sum(predictions.values()) / len(predictions)

    # average std of predictions
    std_ensemble = np.std(list(predictions.values()), axis=0).mean()

    # get evaluation metrics
    mae = mean_absolute_error(y_val, predictions_ensemble)
    r2 = r2_score(y_val, predictions_ensemble)
    rmse = root_mean_squared_error(y_val, predictions_ensemble)

    logger.info(f"Ensemble std: {std_ensemble}")
    logger.info(f"Ensemble MAE: {mae}")
    logger.info(f"Ensemble R2: {r2}")
    logger.info(f"Ensemble RMSE: {rmse}")

    return predictions_ensemble, y_val


def load_model_ensemble(trait: str):
    # list study names
    models = ["mlp", "rf"]
    testsets = [0, 1, 2]
    model_names = [
        f"optuna-debug-{trait}-{model}-testset{testset}"
        for model in models
        for testset in testsets
    ]

    # using glob, get all paths (with varying trial numbers)
    dir_path = os.path.join("data", "train_pipeline", "output", "models", trait)
    model_names_path = {
        name: {
            "pipeline": glob(os.path.join(dir_path, f"model_{name}_*.pkl"))[0],
            "config": glob(os.path.join(dir_path, f"model_{name}_*_config.json"))[0],
            "model_path": os.path.basename(
                glob(os.path.join(dir_path, f"model_{name}_*.pkl"))[0]
            ).removesuffix(".pkl"),
        }
        for name in model_names
    }

    # for model -rf-: add the gee_classifier_path
    for name, path in model_names_path.items():
        if "-rf-" in name:
            path["gee_classifier_path"] = (
                f"projects/ee-speckerfelix/assets/test-models/{path['model_path']}"
            )
        else:
            path["gee_classifier_path"] = None

    # load all models: "model_optuna-debug-{trait}-*.pkl" using pickle_load
    models = {}
    for name, path in model_names_path.items():
        with open(path["pipeline"], "rb") as f:
            with open(path["config"], "r") as f_config:
                models[name] = {
                    "config": json.load(f_config),
                    "pipeline": pickle_load(f),
                    "model_path": path["model_path"],
                    "gee_classifier": (
                        ee.Classifier.load(path["gee_classifier_path"])
                        if path["gee_classifier_path"]
                        else None
                    ),
                }

    return models


def main():
    config = get_config("train_pipeline")
    rerun_and_save_best_optuna_wrapper("lai", config)
    # load_model_ensemble("lai")
    # evaluate_model_ensemble("lai")
    # test_gee_pipeline_predict("lai")


if __name__ == "__main__":
    ee.Initialize()

    # loop over all 6 studies:
    main()
