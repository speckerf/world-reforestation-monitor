"""Train vegetation-specific models using pre-computed 3-fold GroupKFold splits.

Run from the project root:
    python analysis/05_biome_vegetation_specific_models/scripts/03_train_models.py
    python analysis/05_biome_vegetation_specific_models/scripts/03_train_models.py --group 6 --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
import yaml
from loguru import logger
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

BASE_DIR_ALL = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR_ALL))

BASE_DIR_ANALYSIS = Path(__file__).resolve().parents[1]

from train_pipeline.optuna_training import (
    initialize_pipeline,
    optimize_hyperparams,
    predict_and_evaluate,
    prepare_features_and_target,
    report_optuna,
    simulate_data,
)
from train_pipeline.utils_loading import load_grounded_eo_validation_data
from train_pipeline.utils_training import merge_dicts_safe

VEG_CONFIG_PATH = BASE_DIR_ANALYSIS / "config" / "train_pipeline_veg.yaml"
CV_SPLITS_PATH = BASE_DIR_ANALYSIS / "data" / "cv_splits.csv"


def load_veg_config() -> dict:
    with open(VEG_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_study_name(config: dict, group_id: int, fold_id: int) -> str:
    return (
        config["optuna_study_name"]
        .replace("(TRAIT)", config["trait"])
        .replace("(GROUP)", str(group_id))
        .replace("(FOLD)", str(fold_id))
        .replace("(MODEL)", config["model"])
    )


def optuna_init_config_veg(trial, base_config: dict) -> dict:
    """Replicate optuna_init_config logic using provided base_config (no global get_config call)."""
    trait = base_config["trait"]

    if trait in ("fapar", "fcover"):
        transform_target = trial.suggest_categorical(
            "transform_target", ["logit", "log1p", "None"]
        )
    else:
        transform_target = trial.suggest_categorical(
            "transform_target", ["log1p", "standard", "None"]
        )

    config_general = {
        "transform_target": transform_target,
        "nirv_norm": trial.suggest_categorical("nirv_norm", [True, False]),
    }

    config_lut = {
        "num_spectra": 2500 * trial.suggest_int("num_spectra_optuna", 1, 8, log=True),
        "parameter_setup": trial.suggest_categorical(
            "parameter_setup",
            [
                "estevez_2022_mod",
                "foliar_codistribution_mod",
                "kovacs_2023_mod",
                "snap_atbd_mod",
                "wan_2024_lai_mod",
            ],
        ),
        "lai_min": 0.0,
        "lai_max": 15.0,
        "lai_mean": trial.suggest_float("lai_mean", 0.0, 5.0, step=0.2),
        "lai_std": trial.suggest_float("lai_std", 0.2, 5.0, step=0.2),
        "additive_noise": 0.005 * trial.suggest_int("additive_noise_optuna", 1, 7)
        - 0.005,
        "multiplicative_noise": 0.01
        * trial.suggest_int("multiplicative_noise_optuna", 1, 11)
        - 0.01,
        "rsoil_emit_insitu": trial.suggest_categorical(
            "rsoil_emit_insitu", ["emit", "insitu"]
        ),
        "rsoil_fraction": 0.1
        * trial.suggest_int("rsoil_fraction_optuna", 1, 10, log=True),
    }

    config_posthoc = {
        "p_baresoil_insitu": 0.01
        * trial.suggest_int("p_baresoil_insitu_optuna", 1, 11, log=True)
        - 0.01,
        "p_baresoil_s2": 0.01
        * trial.suggest_int("p_baresoil_s2_optuna", 1, 11, log=True)
        - 0.01,
        "p_baresoil_emit": 0.01
        * trial.suggest_int("p_baresoil_emit_optuna", 1, 11, log=True)
        - 0.01,
        "p_urban_s2": 0.005 * trial.suggest_int("p_urban_s2_optuna", 1, 11, log=True)
        - 0.005,
        "p_snowice_s2": 0.005
        * trial.suggest_int("p_snowice_s2_optuna", 1, 11, log=True)
        - 0.005,
    }

    return merge_dicts_safe(config_general, config_posthoc, config_lut)


def prepare_validation_data_for_group(
    df_val: pd.DataFrame,
    cv_splits: pd.DataFrame,
    group_id: int,
    fold_id: int,
    trait: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    splits_group = cv_splits[cv_splits["RECODED_GROUP"] == group_id]
    test_uuids = set(splits_group[splits_group["test_fold"] == fold_id]["uuid"])

    df_group = df_val[df_val["uuid"].isin(splits_group["uuid"])]
    df_val_test = df_group[df_group["uuid"].isin(test_uuids)]
    df_val_train = df_group[~df_group["uuid"].isin(test_uuids)]

    return (
        df_val_train[bands + angles],
        df_val_test[bands + angles],
        df_val_train[[trait]],
        df_val_test[[trait]],
    )


def make_objective(
    base_config: dict,
    df_val: pd.DataFrame,
    cv_splits: pd.DataFrame,
    group_id: int,
    fold_id: int,
):
    def objective(trial) -> float:
        config_optuna = optuna_init_config_veg(trial, base_config)
        config = merge_dicts_safe(dict(base_config), config_optuna)
        trait = config["trait"]
        sim_trait = "lai" if trait == "laie" else trait

        df = simulate_data(config, sim_trait)
        trial.set_user_attr(f"min_simulated_{trait}", float(df[sim_trait].min()))
        trial.set_user_attr(f"max_simulated_{trait}", float(df[sim_trait].max()))

        X, y = prepare_features_and_target(df, sim_trait)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        X_val_train, X_val_test, y_val_train, y_val_test = (
            prepare_validation_data_for_group(
                df_val, cv_splits, group_id, fold_id, trait
            )
        )

        pipeline = initialize_pipeline(config)
        best_model = optimize_hyperparams(pipeline, X_train, y_train, config)

        # increase max iter for MLPRegressor for final model fitting to ensure convergence
        best_model.set_params(regressor__regressor__max_iter=10000)
        model_all = best_model.fit(X, y.squeeze())

        _, scores_sim = predict_and_evaluate(
            best_model, X_train, X_test, y_train, y_test, trait
        )
        _, scores_val = predict_and_evaluate(
            model_all, X_val_train, X_val_test, y_val_train, y_val_test, trait
        )

        scores = {f"sim_{k}": v for k, v in scores_sim.items()}
        scores.update({f"val_{k}": v for k, v in scores_val.items()})

        trial.set_user_attr("config", config)
        trial.set_user_attr("group_id", group_id)
        trial.set_user_attr("fold_id", fold_id)
        report_optuna(scores, trial, trait, config)

        return scores["val_train_rmse"]

    return objective


def run_study(
    base_config: dict,
    df_val: pd.DataFrame,
    cv_splits: pd.DataFrame,
    group_id: int,
    fold_id: int,
    storage: RDBStorage,
) -> None:
    study_name = resolve_study_name(base_config, group_id, fold_id)
    n_trials = base_config["optuna_n_trials"]
    n_startup = base_config["optuna_n_startup_trials"]

    study_exists = study_name in (
        s.study_name for s in optuna.study.get_all_study_summaries(storage=storage)
    )

    sampler = TPESampler(
        seed=None, n_startup_trials=n_startup, constant_liar=True, multivariate=True
    )

    if not study_exists:
        study = optuna.create_study(
            storage=storage, study_name=study_name, sampler=sampler
        )
        logger.info(f"Study '{study_name}' created.")
    else:
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=sampler
        )
        logger.info(
            f"Study '{study_name}' loaded ({len(study.trials)} existing trials)."
        )

    objective = make_objective(base_config, df_val, cv_splits, group_id, fold_id)
    completed_trials = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

    if completed_trials >= n_trials:
        logger.info(
            f"Study '{study_name}' already complete "
            f"({completed_trials}/{n_trials} complete trials)."
        )
        return

    max_trials_callback = MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))
    study.optimize(objective, n_trials=None, callbacks=[max_trials_callback])

    completed_trials = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    logger.info(
        f"Study '{study_name}' finished. "
        f"Completed: {completed_trials}/{n_trials} | Best RMSE: {study.best_value:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vegetation-specific models.")
    parser.add_argument(
        "--trait",
        type=str,
        choices=["fapar", "fcover", "laie"],
        default=None,
        help="Trait to train. If omitted, value from train_pipeline_veg.yaml is used.",
    )
    parser.add_argument(
        "--group", type=int, default=None, help="Run only this recoded group ID."
    )
    parser.add_argument(
        "--fold", type=int, default=None, help="Run only this fold index (0-2)."
    )
    args = parser.parse_args()

    base_config = load_veg_config()
    if args.trait is not None:
        base_config["trait"] = args.trait

    trait = base_config["trait"]

    logger.info(f"Vegetation-specific model training | trait='{trait}'")

    cv_splits = pd.read_csv(CV_SPLITS_PATH)
    cv_splits["RECODED_GROUP"] = cv_splits["RECODED_GROUP"].astype(int)
    cv_splits = cv_splits[cv_splits["test_fold"] >= 0]

    df_val = load_grounded_eo_validation_data().rename(
        columns={"phi": "psi", "sza": "tts", "vza": "tto"}
    )

    storage = RDBStorage(url=base_config["optuna_storage"])

    group_ids = sorted(cv_splits["RECODED_GROUP"].unique().tolist())
    fold_ids = sorted(cv_splits["test_fold"].unique().tolist())

    if args.group is not None:
        group_ids = [args.group]
    if args.fold is not None:
        fold_ids = [args.fold]

    logger.info(f"Groups: {group_ids}  |  Folds: {fold_ids}")
    logger.info(f"Total studies to run: {len(group_ids) * len(fold_ids)}")

    for group_id in group_ids:
        for fold_id in fold_ids:
            logger.info(f"=== Group {group_id}, Fold {fold_id} ===")
            run_study(
                base_config=base_config,
                df_val=df_val,
                cv_splits=cv_splits,
                group_id=group_id,
                fold_id=fold_id,
                storage=storage,
            )


if __name__ == "__main__":
    main()
