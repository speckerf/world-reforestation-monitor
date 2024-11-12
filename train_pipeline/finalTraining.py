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
from train_pipeline.utilsPlotting import plot_predicted_vs_true

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")


def rerun_and_save_best_optuna(config: dict, study=None) -> None:

    if study is None:
        # load the study
        study = optuna.load_study(
            study_name=config["optuna_study_name"], storage=config["optuna_storage"]
        )
    else:
        study = study

    if config["model"] == "mlp":
        trials_filtered = [t for t in study.trials if t.value is not None]

        best_trial_filtered = min(trials_filtered, key=lambda t: t.value)
        best_trial_number = best_trial_filtered.number
        best_trial_value = best_trial_filtered.value
        best_trial_params = best_trial_filtered.params
        # run the best model
        objective(best_trial_filtered, save_model=True)
    else:
        raise ValueError("Only mlp models are supported for now")


def rerun_and_save_best_optuna_wrapper(trait: str, config: dict):
    model = "mlp"
    testsets = [i for i in range(config["group_k_fold_splits"])]

    for testset in testsets:
        study_name = f"optuna-v11-{trait}-{model}-split-{testset}"

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


def check_icos_targets():
    pass
    # check if the accuracy is within 2-sigma thresholds defined by ICOS
    # if trait == "lai":
    #     # threshold: 20% for values >0.5, 0.1 for values <0.5
    #     threshold = 0.5
    #     preds_list = predictions_ensemble.squeeze().tolist()
    #     y_val_list = y_val.squeeze().tolist()

    #     smaller_than_threshold = [
    #         (pred, y_val)
    #         for pred, y_val in zip(preds_list, y_val_list)
    #         if y_val < threshold
    #     ]

    #     greater_than_threshold = [
    #         (pred, y_val)
    #         for pred, y_val in zip(preds_list, y_val_list)
    #         if y_val >= threshold
    #     ]

    #     # check if the accuracy is within 2-sigma thresholds defined by ICOS
    #     smaller_ok = [
    #         True if abs(pred - y_val) < 0.1 else False
    #         for pred, y_val in smaller_than_threshold
    #     ]
    #     greater_ok = [
    #         True if pred / y_val < 1.2 and pred / y_val > 0.8 else False
    #         for pred, y_val in greater_than_threshold
    #     ]

    #     # count percentage of correct predictions
    #     correct = sum(smaller_ok) + sum(greater_ok)
    #     total = len(smaller_than_threshold) + len(greater_than_threshold)
    #     logger.info(
    #         f"Correct predictions: {correct}/{total} ({correct/total*100:.2f}%)"
    #     )


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
        predictions[model_name] = model["pipeline"].predict(X_val)

    # average the predictions across models
    predictions_ensemble = sum(predictions.values()) / len(predictions)

    # average std of predictions
    std_ensemble = np.std(list(predictions.values()), axis=0).mean()

    # get evaluation metrics
    mae = mean_absolute_error(y_val, predictions_ensemble)
    r2 = r2_score(y_val, predictions_ensemble)
    rmse = root_mean_squared_error(y_val, predictions_ensemble)
    me = np.mean(y_val.values - predictions_ensemble)

    logger.info(f"Ensemble std: {std_ensemble}")
    logger.info(f"Ensemble MAE: {mae}")
    logger.info(f"Ensemble R2: {r2}")
    logger.info(f"Ensemble RMSE: {rmse}")

    # stack out-of-sample predictions to get r2_oos

    predictions_oos = {}
    true_values_oos = {}
    validation_data_sites = load_validation_data(return_site=True)[trait]
    for model_name, model in models.items():
        # predict the validation data
        validation_data_sites = load_validation_data(return_site=True)[trait]

        val_ecos_test = model["split"]["val_ecos_test"]

        val_temp = validation_data_sites.loc[
            validation_data_sites["ECO_ID"].isin(val_ecos_test)
        ]
        X_val_temp, y_val_temp = val_temp[features], val_temp[trait]

        predictions_oos_temp = model["pipeline"].predict(X_val_temp)
        predictions_oos[model_name] = predictions_oos_temp.squeeze()
        true_values_oos[model_name] = y_val_temp

    # stack to single array
    predictions_oos_stack = np.concatenate(list(predictions_oos.values()))
    true_values_oos_stack = np.concatenate(list(true_values_oos.values()))

    r2_stacked = r2_score(true_values_oos_stack, predictions_oos_stack)

    # save the metrics to a file
    with open(
        os.path.join(
            "data",
            "train_pipeline",
            "output",
            "models",
            f"metrics_{trait}_ensemble.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "std_ensemble": std_ensemble,
                "mae": mae,
                "r2": r2,
                "rmse": rmse,
                "me": me,
                "r2_stacked": r2_stacked,
            },
            f,
        )

    from train_pipeline.utilsPlotting import plot_predicted_vs_true

    # sample

    plot_predicted_vs_true(
        y_val,
        predictions_ensemble,
        save_plot_filename=os.path.join(
            "data",
            "train_pipeline",
            "output",
            "plots",
            trait,
            f"{'-'.join(model_name.split('-')[0:3])}-ensemble.png",
        ),
        plot_type="density_scatter",
        x_label=f"{trait} - reference measurement",
        y_label=f"{trait} - S2 prediction",
    )

    plot_predicted_vs_true(
        true_values_oos_stack,
        predictions_oos_stack,
        save_plot_filename=os.path.join(
            "data",
            "train_pipeline",
            "output",
            "plots",
            trait,
            f"{'-'.join(model_name.split('-')[0:3])}-stacked_oos.png",
        ),
        plot_type="density_scatter",
        x_label=f"{trait} - reference measurement",
        y_label=f"{trait} - S2 prediction",
    )

    return predictions_ensemble, y_val


def load_model_ensemble(trait: str, models: list[str] = ["mlp"]) -> dict:
    # list study names
    assert models == ["mlp"], "Only mlp models are supported for now"

    testsets = list(range(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["ENSEMBLE_SIZE"]))
    model_names = [
        f"optuna-v11-{trait}-{model}-split-{testset}"
        for model in models
        for testset in testsets
    ]

    # using glob, get all paths (with varying trial numbers)
    dir_path = os.path.join("data", "train_pipeline", "output", "models", trait)
    model_names_path = {
        name: {
            "pipeline": glob(os.path.join(dir_path, f"model_{name}.pkl"))[0],
            "config": glob(os.path.join(dir_path, f"model_{name}_config.json"))[0],
            "model_path": os.path.basename(
                glob(os.path.join(dir_path, f"model_{name}.pkl"))[0]
            ).removesuffix(".pkl"),
            "min_max_bands": glob(
                os.path.join(dir_path, f"min_max_band_values_{name}.json")
            )[0],
            "min_max_label": glob(
                os.path.join(dir_path, f"min_max_label_values_{name}.json")
            )[0],
            "split": os.path.join(dir_path, f"model_{name}_split.json"),
            "df_val_train": os.path.join(dir_path, f"df_val_train_{trait}_{name}.csv"),
            "df_val_test": os.path.join(dir_path, f"df_val_test_{trait}_{name}.csv"),
        }
        for name in model_names
    }

    # load all models: "model_optuna-debug-{trait}-*.pkl" using pickle_load
    models = {}
    for name, path in model_names_path.items():
        with open(path["pipeline"], "rb") as f:
            with open(path["config"], "r") as f_config:
                models[name] = {
                    "config": json.load(f_config),
                    "pipeline": pickle_load(f),
                    "model_path": path["model_path"],
                    "min_max_bands": json.load(open(path["min_max_bands"], "r")),
                    "min_max_label": json.load(open(path["min_max_label"], "r")),
                    "split": json.load(open(path["split"], "r")),
                    # "df_val_train": pd.read_csv(path["df_val_train"]),
                    # "df_val_test": pd.read_csv(path["df_val_test"]),
                }

    return models


def featureToImage(feature):

    properties = feature.toDictionary()
    image = ee.Image.constant(properties.values()).rename(properties.keys())
    return image


def main():
    config = get_config("train_pipeline")
    # rerun_and_save_best_optuna_wrapper("fcover", config)
    # load_model_ensemble("lai")
    evaluate_model_ensemble("fcover")
    # compare_local_gee_rf_predictions("lai")
    # test_gee_pipeline_predict("lai")


if __name__ == "__main__":
    ee.Initialize()

    # loop over all 6 studies:
    main()
