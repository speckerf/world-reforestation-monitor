import copy
import sys
from typing import Self

import numpy as np
from geemap import ml
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rf_get_size_of_string(trees: list[str]) -> dict:
    logger.debug("Calculating size of Random Forest string representation")
    try:
        # Measure the size of the resulting string
        list_size = sys.getsizeof(trees)
        strings_size = sum([sys.getsizeof(s) for s in trees])
        total_size = list_size + strings_size

        sizes = {
            "bytes": total_size,
            "kilobytes": total_size / 1024,
            "megabytes": total_size / 1024 / 1024,
        }
    except Exception as e:
        logger.error(
            f"Error calculating size of Random Forest string representation: {e}"
        )
        sizes = {
            "bytes": 999999999,
            "kilobytes": 999999,
            "megabytes": 999,
        }

    return sizes


def get_model(config):
    if config["model"] == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=config["hidden_layers"],
            activation=config["activation"],
            alpha=config["alpha"],
            learning_rate=config["learning_rate"],
            max_iter=config["max_iter"],
        )
    elif config["model"] == "rf":
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_samples=config["max_samples"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
        )
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    return model


# Custom transformer that converts specified columns (angles) to their cosines
class AngleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary for cosine transformation
        return self

    def transform(self, X):
        # X is a numpy array here, not a DataFrame
        return np.cos(np.deg2rad(X))


class NIRvTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary for cosine transformation
        return self

    def transform(self, X):
        assert X.shape[1] == 10, f"Expected 10 bands, got {X.shape[1]}"
        # problem:
        # X is a numpy array here, not a DataFrame

        # add random number to avoid division by zero
        X = X + np.abs(np.random.rand(*X.shape) * 1e-10)

        NIR = X[["B8"]].values.reshape(-1)
        RED = X[["B4"]].values.reshape(-1)

        # check that NIR + RED is not 0
        if np.any(NIR + RED == 0):
            logger.warning("NIR + RED is 0 for some samples, set to mean sum")
            NIR[NIR + RED == 0] = np.mean(NIR)
            RED[NIR + RED == 0] = np.mean(RED)

        NDVI = (NIR - RED) / (NIR + RED)
        NIRv = NDVI * NIR

        if np.any(NIRv == 0):
            logger.warning(
                "NIRv is 0 for some samples, setting to mean of non-zero values"
            )
            NIRv[NIRv == 0] = np.mean(NIRv[NIRv != 0])

        # divide each row by the corresponding NIRv value
        X_normalized = X / NIRv[:, np.newaxis]
        return X_normalized


# Define the safe logit function
def safe_logit(x):
    # add warning when this clipping was used
    if np.any(x <= 1e-9) or np.any(x >= 1 - 1e-9):
        logger.debug("Clipping values to avoid 0 and 1")
        x = np.clip(x, 1e-9, 1 - 1e-9)  # Clip the values to avoid 0 and 1
    return np.log(x / (1 - x))


# Define the inverse sigmoid function
def safe_inverse_logit(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


def get_pipeline(model: BaseEstimator, config: dict) -> Pipeline:
    angles = ["tts", "tto", "psi"]
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    angle_transformer = Pipeline(steps=[("angle_transformer", AngleTransformer())])
    if config["nirv_norm"]:
        band_transformer = Pipeline(
            steps=[
                ("nirv_transformer", NIRvTransformer()),
                ("scaler", StandardScaler()),
            ],
        )
    else:
        band_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    if config["use_angles_for_prediction"]:
        preprocessor = ColumnTransformer(
            transformers=[
                ("band_transformer", band_transformer, bands),
                ("angle_transformer", angle_transformer, angles),
            ],
            remainder="passthrough",
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[("band_transformer", band_transformer, bands)],
            remainder="passthrough",
        )

    if config["transform_target"] == "log1p":
        regressor = TransformedTargetRegressor(
            regressor=model, func=np.log1p, inverse_func=np.expm1
        )
    elif config["transform_target"] == "standard":
        regressor = TransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
    elif config["transform_target"] == "None":
        regressor = TransformedTargetRegressor(
            regressor=model, func=identity, inverse_func=identity
        )
    elif config["transform_target"] == "logit":
        regressor = TransformedTargetRegressor(
            regressor=model, func=safe_logit, inverse_func=safe_inverse_logit
        )
    else:
        raise ValueError(f"Unknown target transformation: {config['transform_target']}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return pipeline


def merge_dicts_safe(*dicts):
    merged_dict = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in merged_dict:
                logger.error(f"Duplicate key found: {key}")
                raise ValueError(f"Duplicate key found: {key}")
            merged_dict[key] = value
    return merged_dict


def pairwise_nan_remove(X, y):
    mask = np.logical_and(~np.isnan(X).any(axis=1), ~np.isnan(y).any(axis=1))
    if not mask.all():
        logger.warning(f"Removing {np.sum(~mask)} samples with NaNs in either X or y")
    return X[mask], y[mask]


def limit_prediction_range(y_pred, trait):
    min_values = {
        "lai": 0.000,
        "CHL": 0.000,
        "CAR": 0.000,
        "EWT": 0.000,
        "LMA": 0.000,
        "fapar": 0.0001,
        "fcover": 0.0001,
    }
    max_values = {
        "lai": 10.000,
        "CHL": 100.000,
        "CAR": 100.000,
        "EWT": 0.100,
        "LMA": 0.050,
        "fapar": 0.9999,
        "fcover": 0.9999,
    }

    # raise logging message if prediction is out of range
    if np.any(y_pred < min_values[trait]):
        logger.trace(
            f"Prediction below minimum value for {trait}. Setting to minimum value {min_values[trait]}"
        )
    if np.any(y_pred > max_values[trait]):
        logger.trace(
            f"Prediction above maximum value for {trait}. Setting to maximum value {max_values[trait]}"
        )

    y_pred = np.maximum(y_pred, min_values[trait])
    y_pred = np.minimum(y_pred, max_values[trait])
    return y_pred


def r2_score_oos(y_true, y_pred, y_true_train):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    y_true_train = np.array(y_true_train).reshape(-1)
    # Numerator: Residual sum of squares
    numerator = np.sum((y_true - y_pred) ** 2)

    # Denominator: Total sum of squares, using the mean of the training set
    denominator = np.sum((y_true - np.average(y_true_train)) ** 2)

    # Compute the out-of-sample R^2 score
    output_scores = 1 - (numerator / denominator)

    return output_scores
