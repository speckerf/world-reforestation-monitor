from typing import Optional

import ee
import numpy as np
from sklearn.pipeline import Pipeline

from config.config import get_config
from ee_translator.ee_mlp_regressor import eeMLPRegressor
from ee_translator.ee_random_forest_regressor import eeRandomForestRegressor
from ee_translator.ee_standard_scaler import eeStandardScaler
from gee_pipeline.utilsOOD import MinMaxRangeMasker

CONFIG_GEE_PIPELINE = get_config("gee_pipeline")
CONFIG_TRAIN_PIPELINE = get_config("train_pipeline")


# Calculate weights based on linear distance from the midpoint, allowing weights to reach 0
def calculate_linear_weight(
    image: ee.Image, start_date: ee.Date, end_date: ee.Date, total_days: ee.Number
):
    # Calculate the difference in days from the start date.
    days_from_start = image.date().difference(start_date, "day").abs()
    days_to_end = image.date().difference(end_date, "day").abs()

    min_days_to_start_or_end = days_from_start.min(days_to_end)

    weight = min_days_to_start_or_end.divide(total_days.divide(2))
    return image.set("phenology_weight", weight)


def add_random_ensemble_assignment(imgc: ee.ImageCollection) -> ee.ImageCollection:
    return imgc.randomColumn("randomValue", seed=0).map(
        lambda img: img.set(
            "random_ensemble_assignment",
            img.getNumber("randomValue")
            .multiply(CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["ENSEMBLE_SIZE"])
            .toInt8()
            .add(1),
        )
    )


def collapse_to_weighted_mean_and_stddev(imgc: ee.ImageCollection) -> ee.Image:

    # weighted mean
    weighted_sum_pred_img = imgc.map(
        lambda img: img.multiply(img.getNumber("phenology_weight"))
    ).reduce(ee.Reducer.sum())

    weighted_sum_mask_img = imgc.map(
        lambda img: img.mask().toFloat().multiply(img.getNumber("phenology_weight"))
    ).reduce(ee.Reducer.sum())

    weighted_mean = weighted_sum_pred_img.divide(weighted_sum_mask_img)

    # weighted standard deviation
    # Step 1: Calculate the weighted mean: see above
    # Step 2: Calculate the squared deviations from the mean, weighted by the weights
    squared_deviation_img = imgc.map(
        lambda img: img.subtract(weighted_mean)
        .pow(2)
        .multiply(img.getNumber("phenology_weight"))
    ).reduce(ee.Reducer.sum())

    # Step 3: Calculate the variance (sum of weighted squared deviations / sum of weights)
    weighted_variance = squared_deviation_img.divide(weighted_sum_mask_img)

    # Step 4: Take the square root of the variance to get the weighted standard deviation
    weighted_std_dev = weighted_variance.sqrt()

    mean_name = f"{CONFIG_TRAIN_PIPELINE['trait']}_mean"
    std_name = f"{CONFIG_TRAIN_PIPELINE['trait']}_stdDev"
    if CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["CAST_TO_INT16"]:
        # reduce to mean and standard deviation and number of unmasked pixels
        img_mean = (
            weighted_mean.rename(mean_name)
            .multiply(CONFIG_GEE_PIPELINE["INT16_SCALING"][mean_name])
            .toInt16()
        )
        img_std = (
            weighted_std_dev.rename(std_name)
            .multiply(CONFIG_GEE_PIPELINE["INT16_SCALING"][std_name])
            .toInt16()
        )
        img_nobs = imgc.count().rename("n_observations").toUint8()
    else:
        img_mean = weighted_mean.rename(mean_name)
        img_std = weighted_std_dev.rename(std_name)
        img_nobs = imgc.count().rename("n_observations")

    # add bands to single image
    img_to_return = ee.Image([img_mean, img_std, img_nobs])

    return img_to_return


def collapse_to_mean_and_stddev(imgc: ee.ImageCollection) -> ee.Image:
    mean_name = f"{CONFIG_TRAIN_PIPELINE['trait']}_mean"
    std_name = f"{CONFIG_TRAIN_PIPELINE['trait']}_stdDev"
    if CONFIG_GEE_PIPELINE["PIPELINE_PARAMS"]["CAST_TO_INT16"]:
        # reduce to mean and standard deviation and number of unmasked pixels
        img_mean = (
            imgc.mean()
            .rename(mean_name)
            .multiply(CONFIG_GEE_PIPELINE["INT16_SCALING"][mean_name])
            .toInt16()
        )
        img_std = (
            imgc.reduce(ee.Reducer.stdDev())
            .rename(std_name)
            .multiply(CONFIG_GEE_PIPELINE["INT16_SCALING"][std_name])
            .toInt16()
        )
        img_nobs = imgc.count().rename("n_observations").toUint8()
    else:
        img_mean = imgc.mean().rename(mean_name)
        img_std = imgc.reduce(ee.Reducer.stdDev()).rename(std_name)
        img_nobs = imgc.count().rename("n_observations")

    img_to_return = ee.Image([img_mean, img_std, img_nobs])
    return img_to_return


def ee_nirv_normalisation(image: ee.Image):
    reflectance_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    NDVI = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    NIRv = NDVI.multiply(image.select("B8")).rename("NIRv")

    image_normalized = image.select(reflectance_bands).divide(NIRv)
    image_to_return = image.addBands(image_normalized, overwrite=True)
    return image_to_return


def ee_angle_transformer(image: ee.Image):
    # cosine transformation of angles
    image_angles = image.select(["tts", "tto", "psi"]).multiply(np.pi / 180).cos()
    image_to_return = image.addBands(image_angles, overwrite=True)
    return image_to_return


def ee_logit_transform(image: ee.Image, trait: str):
    # logit transformation of trait
    #  np.log(x / (1 - x))
    image = image.addBands(
        image.select(trait).log().divide(image.select(trait).subtract(1))
    )
    return image


def ee_logit_inverse_transform(image: ee.Image, trait: str):
    # inverse logit transformation of trait
    # 1 / (1 + np.exp(-x))
    image = image.addBands(
        image.select(trait).exp().divide(image.select(trait).exp().add(1))
    )
    return image


def ee_log1p_inverse_transform(image: ee.Image, trait: str):
    # inverse log1p transformation of trait
    # np.exp(x) - 1
    image = image.select(trait).exp().subtract(1)
    return image


def eePipelinePredictMap(
    pipeline: Pipeline,
    imgc: ee.ImageCollection,
    trait: str,
    model_config: dict,
    gee_random_forest: Optional[ee.Classifier] = None,
    min_max_bands: Optional[dict] = None,
    min_max_label: Optional[dict] = None,
):
    # get the bands and angles
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    # convert reflectances from int to float by dividing by 10000
    imgc = imgc.map(
        lambda image: image.addBands(image.select(bands).divide(10000), overwrite=True)
    )

    # mask all pixels with reflectance values outside of the min_max reflectance values
    if min_max_bands is not None:
        min_max_band_masker = MinMaxRangeMasker(min_max_bands)
        imgc = imgc.map(min_max_band_masker.ee_mask)

    if model_config["nirv_norm"]:
        imgc = imgc.map(ee_nirv_normalisation)
    if model_config["use_angles_for_prediction"]:
        features = bands + angles
        imgc = imgc.map(ee_angle_transformer)
        imgc = imgc.select(features)
    else:
        features = bands
        imgc = imgc.select(features)

    # always apply standard scaler
    band_scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["band_transformer"]
        .named_steps["scaler"]
    )

    ee_band_scaler = eeStandardScaler(band_scaler)
    # a = ee_band_scaler.transform_image(imgc.first())
    imgc = imgc.map(ee_band_scaler.transform_image)

    # apply model:
    if model_config["model"] == "mlp":
        # IMPORTANT: .regressor_ refers to the actual model, while .regressor only refers to the untrained model
        ee_model = eeMLPRegressor(
            pipeline.named_steps["regressor"].regressor_, trait_name=trait
        )
    elif model_config["model"] == "rf":
        ee_model = eeRandomForestRegressor(
            feature_names=features, trait_name=trait, ee_rf_model=gee_random_forest
        )
    imgc = imgc.map(lambda image: ee_model.predict(image))

    # apply inverse transformations
    if model_config["transform_target"] == "log1p":
        imgc = imgc.map(
            lambda image: ee_log1p_inverse_transform(image, trait).copyProperties(image)
        )
    elif model_config["transform_target"] == "logit":
        imgc = imgc.map(
            lambda image: ee_logit_inverse_transform(image, trait).copyProperties(image)
        )
    elif model_config["transform_target"] == "standard":
        target_scaler = pipeline.named_steps["regressor"].transformer_
        target_ee_scaler = eeStandardScaler(
            target_scaler, feature_names=[trait]
        )  # must be a list
        imgc = imgc.map(
            lambda image: target_ee_scaler.inverse_transform_column(
                image, trait
            ).copyProperties(image)
        )
    elif model_config["transform_target"] == "None":
        imgc = imgc
    else:
        raise ValueError(
            f"Unknown target transformation: {model_config['transform_target']}"
        )

    if min_max_label is not None:
        min_max_label_masker = MinMaxRangeMasker(min_max_label)
        imgc = imgc.map(min_max_label_masker.ee_mask)

    return imgc


# def eePipelinePredict(
#     pipeline: Pipeline, image: ee.Image, trait: str, trial_config_dict: dict
# ):
#     bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
#     angles = ["tts", "tto", "psi"]

#     if trial_config_dict["nirv_norm"]:
#         image = ee_nirv_normalisation(image)
#     if trial_config_dict["use_angles_for_prediction"]:
#         features = bands + angles
#         image = ee_angle_transformer(image)
#         image = image.select(features)
#     else:
#         features = bands
#         image = image.select(bands)

#     # always apply standard scaler
#     band_scaler = (
#         pipeline.named_steps["preprocessor"]
#         .named_transformers_["band_transformer"]
#         .named_steps["scaler"]
#     )
#     # TODO: import eeStandardScaler
#     ee_band_scaler = eeStandardScaler(band_scaler)
#     image = ee_band_scaler.transform_image(image)

#     # apply model:
#     if trial_config_dict["model"] == "mlp":
#         # IMPORTANT: .regressor_ refers to the actual model, while .regressor only refers to the untrained model
#         ee_model = eeMLPRegressor(
#             pipeline.named_steps["regressor"].regressor_, trait_name=trait
#         )
#     elif trial_config_dict["model"] == "rf":
#         ee_model = eeRandomForestRegressor(
#             pipeline.named_steps["regressor"].regressor_,
#             feature_names=features,
#             trait_name=trait,
#         )

#     image = ee_model.predict(image)

#     # apply inverse transformations
#     if trial_config_dict["transform_target"] == "log1p":
#         image = ee_log1p_inverse_transform(image, trait)
#     elif trial_config_dict["transform_target"] == "logit":
#         image = ee_logit_inverse_transform(image, trait)
#     elif trial_config_dict["transform_target"] == "standard":
#         target_scaler = pipeline.named_steps["regressor"].transformer_
#         target_ee_scaler = eeStandardScaler(
#             target_scaler, feature_names=[trait]
#         )  # must be a list
#         image = target_ee_scaler.inverse_transform_column(image, trait)
#     elif trial_config_dict["transform_target"] == "None":
#         image = image
#     else:
#         raise ValueError(
#             f"Unknown target transformation: {trial_config_dict['transform_target']}"
#         )

#     return image
