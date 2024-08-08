import copy
from typing import Self

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
        )
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    return model


class EcoregionSpecificModel(BaseEstimator, RegressorMixin):
    def __init__(
        self, pipeline: Pipeline, config: dict, fit_per_ecoregion: bool = True
    ):
        assert fit_per_ecoregion, "Only fit_per_ecoregion=True is supported"
        self.pipeline: Pipeline = pipeline
        self.config: dict = config
        self.fit_per_ecoregion: bool = fit_per_ecoregion
        self.ecoregions_: list[str] = []
        self.per_ecoregion_pipeline_: dict[str, Pipeline] = {}

    def fit(
        self, X: dict[str, np.ndarray], y: dict[str, np.ndarray], ecoregions: list[str]
    ) -> Self:
        assert self.fit_per_ecoregion, "Only fit_per_ecoregion=True is supported"
        # X needs to be a dictionary with eco_id as key and the corresponding data as value
        self.ecoregions_ = list(np.unique(ecoregions))
        self.per_ecoregion_pipeline_ = {
            ecoregion: copy.deepcopy(self.pipeline) for ecoregion in self.ecoregions_
        }
        for eco_id, pipeline in self.per_ecoregion_pipeline_.items():
            X_ecoregion = X[eco_id]
            y_ecoregion = y[eco_id]
            X_ecoregion, y_ecoregion = pairwise_nan_remove(X_ecoregion, y_ecoregion)

            pipeline.fit(X_ecoregion, y_ecoregion)

        return self

    def predict(self, X: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        assert all(
            eco_id in self.ecoregions_ for eco_id in X.keys()
        ), "Some ecoregions are missing in the fitted model"
        assert all(
            eco_id in self.per_ecoregion_pipeline_.keys() for eco_id in X.keys()
        ), "Some ecoregions are missing in the fitted model"
        return {
            eco_id: pipeline.predict(X[eco_id])
            for eco_id, pipeline in self.per_ecoregion_pipeline_.items()
            if eco_id in X.keys()
        }


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
        # X is a numpy array here, not a DataFrame

        # add random number to avoid division by zero
        X = X + np.abs(np.random.rand(*X.shape) * 1e-10)

        NIR = X[["B8"]].values.squeeze()
        RED = X[["B4"]].values.squeeze()

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


def get_pipeline(model: BaseEstimator, config: dict) -> Pipeline:
    angles = ["tts", "tto", "psi"]
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    angle_transformer = Pipeline(steps=[("angle_transformer", AngleTransformer())])
    if config["nirv_norm"]:
        band_transformer = Pipeline(
            steps=[
                ("nirv_transformer", NIRvTransformer()),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        band_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    if config["use_angles_for_prediction"]:
        preprocessor = ColumnTransformer(
            transformers=[
                ("angle_transformer", angle_transformer, angles),
                ("band_transformer", band_transformer, bands),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[("band_transformer", band_transformer, bands)]
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
            regressor=model, func=lambda x: x, inverse_func=lambda x: x
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
    }
    max_values = {
        "lai": 10.000,
        "CHL": 100.000,
        "CAR": 100.000,
        "EWT": 0.100,
        "LMA": 0.050,
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
