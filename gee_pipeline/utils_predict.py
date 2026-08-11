from typing import Any, Literal

import ee
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ee_translator.ee_mlp_regressor import eeMLPRegressor
from ee_translator.ee_standard_scaler import eeStandardScaler
from gee_pipeline.utils_ood import MinMaxRangeMasker
from gee_pipeline.utils_tiles import add_group


def ee_calibrate_std(
    image: ee.Image,
    recalibration_table: pd.DataFrame,
    variable: Literal["laie", "fcover", "fapar"],
) -> ee.Image:
    assert all(col in recalibration_table.columns for col in ["y_pred", "tau"]), (
        "Calibration table must contain 'pred_mean' and 'tau' columns"
    )

    mean_bandname = f"{variable}_mean"
    std_bandname = f"{variable}_stdDev"

    mean_image = image.select(mean_bandname)
    std_image = image.select(std_bandname)

    y_pred_inter = ee.List(recalibration_table["y_pred"].values.tolist())
    tau_inter = ee.List(recalibration_table["tau"].values.tolist())

    # now we want to get the interpolated tau value for each pixel based on its predicted mean value
    tau_image = mean_image.interpolate(
        y_pred_inter,
        tau_inter,
        behavior="clamp",  # BUG: I misunderstood "extrapolate": resulted in unrealistic recalibration if prediciton outside the realistic domain: changed here to "clamp" --> plus we need to change to mask before rescaling!!!!
    ).rename("tau")

    # use ee.Image.interpolate to recalibrate std_image / values outside of the range are copied (behavior='input')
    std_image_calibrated = std_image.multiply(tau_image).rename(std_bandname)

    return image.addBands(std_image_calibrated, overwrite=True)


# Calculate weights based on linear distance from the midpoint, allowing weights to reach 0
def calculate_linear_weight(
    image: ee.Image, start_date: ee.Date, end_date: ee.Date, total_days: ee.Number
):
    raise DeprecationWarning("This function is deprecated.")
    # Calculate the difference in days from the start date.
    days_from_start = image.date().difference(start_date, "day").abs()
    days_to_end = image.date().difference(end_date, "day").abs()

    min_days_to_start_or_end = days_from_start.min(days_to_end)

    weight = min_days_to_start_or_end.divide(total_days.divide(2))
    return image.set("phenology_weight", weight)


def add_random_ensemble_assignment(
    imgc: ee.ImageCollection, ensemble_size: int
) -> ee.ImageCollection:
    raise DeprecationWarning(
        "This function is deprecated and should not be used anymore."
    )
    return imgc.randomColumn("randomValue", seed=0).map(
        lambda img: img.set(
            "random_ensemble_assignment",
            img.getNumber("randomValue").multiply(ensemble_size).floor().add(1),
        )
    )


def add_random_balanced_ensemble_assignment(
    imgc: ee.ImageCollection, ensemble_size: int
):
    raise DeprecationWarning(
        "This function is deprecated and should not be used anymore."
    )
    size = imgc.size()

    # add group key
    imgc = imgc.map(add_group)  # with property name 'group'

    sorted_imgc = imgc.sort("group")

    imgs = sorted_imgc.toList(size)
    idxs = ee.List.sequence(0, size.subtract(1))

    return ee.ImageCollection(
        idxs.zip(imgs).map(
            lambda pair: ee.Image(ee.List(pair).get(1)).set(
                "random_ensemble_assignment",
                ee.Number(ee.List(pair).get(0)).mod(ensemble_size).add(1),
            )
        )
    )


def collapse_to_mean_and_stddev(
    imgc: ee.ImageCollection,
    return_count: bool,
    variable: Literal["laie", "fcover", "fapar"],
    clamp_range: tuple[int, int] | None = None,
) -> ee.Image:
    raise DeprecationWarning(
        "This function is deprecated and should not be used anymore."
    )
    mean_name = f"{variable}_mean"
    std_name = f"{variable}_stdDev"

    img_mean = imgc.mean().rename(mean_name)
    img_std = imgc.reduce(ee.Reducer.sampleStdDev()).rename(
        std_name
    )  # use sampleStdDev
    if return_count:
        img_nobs = imgc.reduce(ee.Reducer.count()).rename(f"{variable}_count")

    # clamp predictions
    if clamp_range is not None:
        img_mean = img_mean.clamp(clamp_range[0], clamp_range[1]).copyProperties(
            img_mean
        )

    if return_count:
        img_to_return = ee.Image([img_mean, img_std, img_nobs])
    else:
        img_to_return = ee.Image([img_mean, img_std])
    return img_to_return


def scale_and_cast_to_int(
    image: ee.Image,
    variable: Literal["laie", "fcover", "fapar"],
    mean_scaling: int,
    std_scaling: int,
) -> ee.Image:
    mean_name = f"{variable}_mean"
    std_name = f"{variable}_stdDev"

    mean_image = image.select([mean_name]).multiply(mean_scaling).toInt16()
    std_image = image.select([std_name]).multiply(std_scaling).toInt16()
    count_image = image.select([f"{variable}_count"]).toUint8()
    return_image = ee.Image([mean_image, std_image, count_image])

    if f"{variable}_stdDev_across" in image.bandNames().getInfo():
        std_across_image = (
            image.select([f"{variable}_stdDev_across"]).multiply(std_scaling).toInt16()
        )
        return_image = return_image.addBands(std_across_image)

    if f"{variable}_stdDev_within" in image.bandNames().getInfo():
        std_within_image = (
            image.select([f"{variable}_stdDev_within"]).multiply(std_scaling).toInt16()
        )
        return_image = return_image.addBands(std_within_image)

    return return_image


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
    return (
        image.select(trait).expression("1 / (1 + exp(-x))", {"x": image}).rename(trait)
    )


def ee_log1p_inverse_transform(image: ee.Image, trait: str):
    # inverse log1p transformation of trait
    # np.exp(x) - 1
    return image.select(trait).exp().subtract(1).rename(trait)


def eePipelinePredictMap(
    pipeline: Pipeline,
    imgc: ee.ImageCollection,
    trait: str,
    model_config: dict,
    min_max_bands: dict | None = None,
    min_max_label: dict | None = None,
):
    raise DeprecationWarning(
        "This function is deprecated and should not be used anymore. Use eeEnsemblePredictSingleImg instead."
    )
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

    features = bands + angles
    imgc = imgc.map(ee_angle_transformer)
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
    else:
        raise ValueError("Only mlp models are supported for now")
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
        pass
    else:
        raise ValueError(
            f"Unknown target transformation: {model_config['transform_target']}"
        )

    if min_max_label is not None:
        min_max_label_masker = MinMaxRangeMasker(min_max_label)
        imgc = imgc.map(min_max_label_masker.ee_mask)

    return imgc


def _preprocess_base(x: ee.Image) -> ee.Image:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]
    features = bands + angles

    # Convert reflectances
    x = x.addBands(x.select(bands).divide(10000), overwrite=True)

    # Angles + feature selection
    x = ee_angle_transformer(x)
    return x.select(features)


def _apply_band_scaler(x: ee.Image, pipeline) -> ee.Image:
    # Always apply standard scaler
    band_scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["band_transformer"]
        .named_steps["scaler"]
    )
    ee_band_scaler = eeStandardScaler(band_scaler)
    return ee_band_scaler.transform_image(x)


def _predict_regressor(x: ee.Image, pipeline, trait_name: str) -> ee.Image:
    # IMPORTANT: .regressor_ refers to the trained model
    ee_model = eeMLPRegressor(
        pipeline.named_steps["regressor"].regressor_, trait_name=trait_name
    )
    return ee_model.predict(x)


def _inverse_target_transform(
    img_pred: ee.Image,
    pipeline,
    cfg: dict[str, Any],
    trait_name: str,
) -> ee.Image:
    t = cfg.get("transform_target")
    if t in (None, "None"):
        return img_pred

    if t == "log1p":
        return ee.Image(ee_log1p_inverse_transform(img_pred, trait_name))

    if t == "logit":
        return ee.Image(ee_logit_inverse_transform(img_pred, trait_name))

    if t == "standard":
        target_scaler = pipeline.named_steps["regressor"].transformer_
        target_ee_scaler = eeStandardScaler(
            target_scaler, feature_names=[trait_name]
        )  # must be list
        return ee.Image(target_ee_scaler.inverse_transform_column(img_pred, trait_name))

    raise ValueError(f"Unknown target transformation: {t}")


def _predict_one_member(
    img_base: ee.Image, model: dict[str, Any], trait_name: str
) -> ee.Image:
    pipeline = model["pipeline"]
    cfg = model.get("config", {})

    # Start from base image for this member (fixes img_temp-before-assignment bug)
    x = img_base

    # Optional per-model masking
    mm = model.get("min_max_bands")
    if mm is not None:
        x = MinMaxRangeMasker(mm).ee_mask(x)

    # Optional per-model NIRv normalisation
    if cfg.get("nirv_norm", False):
        x = ee_nirv_normalisation(x)

    # Scaling + prediction
    x = _apply_band_scaler(x, pipeline)
    y = _predict_regressor(x, pipeline, trait_name)

    # Inverse target transform
    return _inverse_target_transform(y, pipeline, cfg, trait_name)


def eeEnsemblePredictSingleImg(
    ensemble,
    img: ee.Image,
    variable: Literal["laie", "fcover", "fapar"],
    mask_and_clamp: bool = True,
    mask_range: tuple[float, float] | None = None,
    clamp_range: tuple[float, float] | None = None,
    calibrate_uncertainty: bool | None = False,
    uncertainty_calibration_table: pd.DataFrame | None = None,
) -> ee.Image:
    """
    Predicts using an ensemble of models on a single image (ee.ImageCollection with one image)
    and returns the mean and stdDev of the predictions as an ee.Image.
    """
    mean_bandname = f"{variable}_mean"
    std_bandname = f"{variable}_stdDev"

    if mask_and_clamp:
        assert mask_range is not None and clamp_range is not None, (
            "mask_range and clamp_range must be provided if mask_and_clamp is True"
        )

    # ------------------------------------------------------------------
    # shared preprocessing (identical for all ensemble members)
    # ------------------------------------------------------------------
    img_preprocessed = _preprocess_base(img)

    preds = []
    for model_name, model in ensemble.items():
        try:
            preds.append(_predict_one_member(img_preprocessed, model, variable))
        except Exception as e:
            raise RuntimeError(f"Ensemble member '{model_name}' failed.") from e

    preds_ic = ee.ImageCollection(preds)
    preds_mean = preds_ic.mean().rename(mean_bandname)
    preds_std = preds_ic.reduce(ee.Reducer.sampleStdDev()).rename(std_bandname)

    # mask predictions:
    # previous bug: unrealistic predicitons where not masked by default anymore; lead in combination with extrapolate recalibration (negative reclalibration was possible due to that) to undesired behaviour.
    if mask_and_clamp:
        # first mask:
        mask = preds_mean.gte(mask_range[0]).And(preds_mean.lte(mask_range[1]))
        preds_mean = preds_mean.updateMask(mask)
        preds_std = preds_std.updateMask(mask)

        # then clamp:
        preds_mean = preds_mean.clamp(clamp_range[0], clamp_range[1])

    if calibrate_uncertainty:
        if uncertainty_calibration_table is None:
            raise ValueError(
                "uncertainty_calibration_table must be provided when recalibrate_uncertainty is True"
            )
        return ee_calibrate_std(
            ee.Image([preds_mean, preds_std]),
            uncertainty_calibration_table,
            variable=variable,
        )
    else:
        return ee.Image([preds_mean, preds_std])


# test ensemble assignment
if __name__ == "__main__":
    ee.Initialize()
