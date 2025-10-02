import json
import os
from pickle import dump as pickle_dump

import ee
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, HalvingGridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from config.config import get_config
from rtm_pipeline_python.classes import (
    helper_apply_posthoc_modifications,
    rtm_simulator,
)
from train_pipeline.utilsLoading import load_grounded_eo_validation_data
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


def save_lut_and_ranges(
    df, trait, save_folder, model_domain, study_name, bands
) -> None:
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
    if trait == "laie":
        min_max_label_values = {"lai": {"min": min(df["lai"]), "max": max(df["lai"])}}
    else:
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
            f"model_domain_codes_{study_name}.json",
        ),
        "w",
    ) as f:
        json.dump(model_domain, f)

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


def initialize_configuration(trial):
    config_optuna = optuna_init_config(trial)
    config_general = get_config("train_pipeline")
    return merge_dicts_safe(config_general, config_optuna)


def simulate_data(config, trait):
    lut_simulator = rtm_simulator(
        config, os.path.join("rtm_pipeline_R", "src", "run_prosail.R")
    )
    df = lut_simulator.generate_lut()
    df = helper_apply_posthoc_modifications(df, trait, config)
    return df


def prepare_features_and_target(df, trait):
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]
    feature_names = bands + angles
    target = [trait]
    return df[feature_names], df[target]


def prepare_validation_data(config, trait, trial):
    # Load and filter validation data
    df_val_trait = load_grounded_eo_validation_data().rename(
        columns={"phi": "psi", "sza": "tts", "vza": "tto"}
    )

    all_eco_ids = df_val_trait["ECO_ID"].unique()

    # GroupKFold splits
    skf = GroupKFold(n_splits=config["group_k_fold_splits"])
    splits = list(skf.split(df_val_trait, groups=df_val_trait["ECO_ID"]))

    val_ecos_train = [
        df_val_trait["ECO_ID"].values[df_val_train_idx].unique()
        for df_val_train_idx, _ in splits
    ]

    val_ecos_test = [
        df_val_trait["ECO_ID"].values[df_val_test_idx].unique()
        for _, df_val_test_idx in splits
    ]
    #

    # Get indices for the current train/test split
    val_eco_train_split_indices, val_eco_test_split_indices = splits[
        config["group_k_fold_current_split"]
    ]

    log_splits(splits, df_val_trait, current_fold=config["group_k_fold_current_split"])

    df_val_train_current = df_val_trait.iloc[val_eco_train_split_indices]
    df_val_test_current = df_val_trait.iloc[val_eco_test_split_indices]

    # Create X and y for train and test
    target = [trait]

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    X_val_train, y_val_train = (
        df_val_train_current[bands + angles],
        df_val_train_current[target],
    )
    X_val_test, y_val_test = (
        df_val_test_current[bands + angles],
        df_val_test_current[target],
    )

    # report ecoregions splits to optuna
    trial.set_user_attr(
        "val_ecos_train",
        val_ecos_train[config["group_k_fold_current_split"]].astype(int).tolist(),
    )
    trial.set_user_attr(
        "val_ecos_test",
        val_ecos_test[config["group_k_fold_current_split"]].astype(int).tolist(),
    )

    return X_val_train, X_val_test, y_val_train, y_val_test


def initialize_pipeline(config):
    model = get_model(config)
    pipeline = get_pipeline(model, config)
    return pipeline


def optimize_hyperparams(pipeline, X_train, y_train, config):
    logger.debug(f"Hyperparameter optimization using HalvingGridSearchCV...")
    param_grid = config["mlp_grid"].copy()

    hidden_layers_dict = {
        "5": (5,),
        "10": (10,),
        "5_5": (5, 5),
        "10_5": (10, 5),
        "10_10": (10, 10),
    }

    param_grid["hidden_layer_sizes"] = [
        hidden_layers_dict[hidden_layer]
        for hidden_layer in param_grid["hidden_layer_sizes"]
    ]
    param_grid_pipeline = {
        f"regressor__regressor__{key}": value for key, value in param_grid.items()
    }

    halving_search = HalvingGridSearchCV(
        pipeline,
        param_grid_pipeline,
        cv=3,
        random_state=42,
        n_jobs=1,
        scoring="neg_root_mean_squared_error",
        factor=3,
        verbose=1,
    )
    halving_search.fit(X_train, y_train)
    return halving_search.best_estimator_


def predict_and_evaluate(model, X_train, X_test, y_train, y_test, trait):
    if "uuid" in X_train.columns:
        # assert that the uuids are in the same order
        assert (X_train["uuid"].values == y_train["uuid"].values).all()
        assert (X_test["uuid"].values == y_test["uuid"].values).all()

        X_train = X_train.drop(columns="uuid")
        X_test = X_test.drop(columns="uuid")
        y_train = y_train.drop(columns="uuid")
        y_test = y_test.drop(columns="uuid")

    predictions = {
        "y_train_pred": limit_prediction_range(model.predict(X_train), trait),
        "y_test_pred": limit_prediction_range(model.predict(X_test), trait),
    }

    scores = {
        "train_rmse": root_mean_squared_error(y_train, predictions["y_train_pred"]),
        "test_rmse": root_mean_squared_error(y_test, predictions["y_test_pred"]),
        "train_mae": mean_absolute_error(y_train, predictions["y_train_pred"]),
        "test_mae": mean_absolute_error(y_test, predictions["y_test_pred"]),
        "train_me": np.mean(y_train - predictions["y_train_pred"]),
        "test_me": np.mean(y_test - predictions["y_test_pred"]),
        "train_r2": r2_score(y_train, predictions["y_train_pred"]),
        "test_r2": r2_score(y_test, predictions["y_test_pred"]),
        "test_r2_oos": r2_score_oos(
            y_test, predictions["y_test_pred"], y_true_train=y_train
        ),
    }
    return predictions, scores


def save_model_and_evaluation(model, eval_metrics, save_folder, trial, config):
    os.makedirs(save_folder, exist_ok=True)
    model_filename = f"model_{trial.user_attrs['config']['optuna_study_name']}"
    full_model_path = os.path.join(save_folder, f"{model_filename}.pkl")
    # Save the model, metrics, and configuration
    with open(full_model_path, "wb") as f:
        pickle_dump(model, f)
    with open(os.path.join(save_folder, f"{model_filename}.json"), "w") as f:
        json.dump(eval_metrics, f)
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


def report_optuna(scores, trial, trait, config):
    for key, val in scores.items():
        if "_rmse" in key:
            trial.set_user_attr(key, min(val, config["optuna_report_thresh"][trait]))
        elif "_mae" in key:
            trial.set_user_attr(key, min(val, config["optuna_report_thresh"][trait]))
        elif "_r2" in key:
            trial.set_user_attr(key, max(val, -1))
        elif "_me" in key:
            trial.set_user_attr(key, val)

        # other parameters to report
        trial.set_user_attr("config", config)
    return None


def create_model_domain_codes(df):
    """
    df: DataFrame with band values (prosail simulations + added spectra (baresoil, snow, etc...))
    returns: list with 10 digit domain codes (from 1-9: 0.1 reflectance steps --> similar so SL2P)
    """
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    assert all(band in df.columns for band in bands), "DataFrame must contain all bands"

    X = df[bands].copy().to_numpy()  # (n_samples, 10)

    powers_of_ten = np.array([10**i for i in range(len(bands))]).reshape(
        1, -1
    )  # (1, 10)

    domain_codes = np.sum(np.uint8(np.ceil(X * 10) % 10) * powers_of_ten, axis=1)

    domain_codes_unique = np.unique(domain_codes.astype(np.uint32))
    return domain_codes_unique.tolist()


def objective(trial, save_model=False):
    config = initialize_configuration(trial)
    trait = config["trait"]

    # Data simulation
    if trait == "laie":
        df = simulate_data(config, "lai")
        trial.set_user_attr(f"min_simulated_{trait}", min(df["lai"]))
        trial.set_user_attr(f"max_simulated_{trait}", max(df["lai"]))
    else:
        df = simulate_data(config, trait)
        trial.set_user_attr(f"min_simulated_{trait}", min(df[trait]))
        trial.set_user_attr(f"max_simulated_{trait}", max(df[trait]))

    if save_model:
        save_folder = os.path.join(
            "data",
            "train_pipeline",
            "output",
            "models",
            trait,
        )
        study_name = trial.user_attrs["config"]["optuna_study_name"]

        model_domain = create_model_domain_codes(df)

        save_lut_and_ranges(
            df=df,
            trait=trait,
            save_folder=save_folder,
            model_domain=model_domain,
            study_name=study_name,
            bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        )

    if trait == "laie":
        X, y = prepare_features_and_target(df, "lai")
    else:
        X, y = prepare_features_and_target(df, trait)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Validation data preparation
    X_val_train, X_val_test, y_val_train, y_val_test = prepare_validation_data(
        config=config, trait=trait, trial=trial
    )

    # Model training
    pipeline = initialize_pipeline(config)
    best_model = optimize_hyperparams(pipeline, X_train, y_train, config)
    logger.debug(f"Refit model with all simulated training data...")
    model_train = best_model.fit(X_train, y_train)
    model_all = best_model.fit(X, y)

    # Prediction and evaluation
    predictions_sim, scores_sim = predict_and_evaluate(
        model_train, X_train, X_test, y_train, y_test, trait
    )

    # Predict validation data
    predictions_val, scores_val = predict_and_evaluate(
        model_all, X_val_train, X_val_test, y_val_train, y_val_test, trait
    )

    # concat both scores: appending 'sim' and 'val' to keys
    scores = {f"sim_{key}": val for key, val in scores_sim.items()}
    scores.update({f"val_{key}": val for key, val in scores_val.items()})

    # Save model and evaluation (optional)
    if save_model:
        save_folder = os.path.join("data", "train_pipeline", "output", "models", trait)
        save_model_and_evaluation(best_model, scores, save_folder, trial, config)

    # report scores to optuna
    if not save_model:
        report_optuna(scores, trial, trait, config)

    # plotting
    if save_model:
        model_filename = f"model_{trial.user_attrs['config']['optuna_study_name']}"
        plot_predicted_vs_true(
            y_true=y_train,
            y_pred=predictions_sim["y_train_pred"],
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_sim_train_{model_filename}.png"
            ),
        )

        plot_predicted_vs_true(
            y_true=y_test,
            y_pred=predictions_sim["y_test_pred"],
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_sim_test_{model_filename}.png"
            ),
        )

        plot_predicted_vs_true(
            y_true=y_val_train[trait],
            y_pred=predictions_val["y_train_pred"],
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_val_train_{model_filename}.png"
            ),
        )

        plot_predicted_vs_true(
            y_true=y_val_test[trait],
            y_pred=predictions_val["y_test_pred"],
            plot_type="density_scatter",
            save_plot_filename=os.path.join(
                save_folder, f"predicted_vs_true_val_test_{model_filename}.png"
            ),
        )

    return scores["val_train_rmse"]


def make_stop_callback(total_trials):
    def stop_when_reached(study, trial):
        if len(study.trials) >= total_trials:
            study.stop()

    return stop_when_reached


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

    N_TOTAL_TRIALS = config_general["optuna_n_trials"]
    N_STARTUP_TRIALS = config_general["optuna_n_startup_trials"]

    stop_cb = make_stop_callback(N_TOTAL_TRIALS)

    # Create the study if it doesn't exist
    if not study_exists:
        sampler = TPESampler(
            seed=None,
            n_startup_trials=N_STARTUP_TRIALS,
            constant_liar=True,
            multivariate=True,
        )
        study = optuna.create_study(
            storage=storage, study_name=study_name, sampler=sampler
        )
        logger.info(f"Study '{study_name}' created.")
    else:
        sampler = TPESampler(
            seed=None,
            n_startup_trials=N_STARTUP_TRIALS,
            constant_liar=True,
            multivariate=True,
        )
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=sampler
        )
        logger.info(f"Study '{study_name}' loaded.")

    study.optimize(objective, n_trials=None, callbacks=[stop_cb])

    # get the best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")


if __name__ == "__main__":
    main()
