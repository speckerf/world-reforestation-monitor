import re

# assess goodness of fit for all models
from math import sqrt

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from SL2P_PYTHON.felix_model import load_SL2P_model
from test_groundedeo.predict_grounded_eo import load_model as load_grounded_eo_model
from train_pipeline.finalTraining import (
    load_model_ensemble as load_specker_model_ensemble,
)
from train_pipeline.utilsLoading import load_grounded_eo_validation_data
from train_pipeline.utilsTraining import r2_score_oos, uncertainty_agreement_ratio


def calculate_metrics(y_true, y_pred, model_name):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # pearson correlation
    pearson_r, _ = pearsonr(y_true, y_pred)

    logger.info(
        f"{model_name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}, Pearson r: {pearson_r:.3f}"
    )
    return {
        "model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson_r": pearson_r,
    }


def predict_grounded_eo() -> pd.DataFrame:
    """Predict LAI and FAPAR with Grounded EO models; returns columns:
    grounded_lai_mean, grounded_lai_std, grounded_fapar_mean, grounded_fapar_std.
    Predictions are clipped to [0, upper_lim] where applicable.
    """
    val = load_grounded_eo_validation_data()
    out = pd.DataFrame(index=val.index)

    # Precompute cosine angles if raw angles exist and cosines not already present
    df = val.copy()
    if {"sza", "vza", "phi"}.issubset(df.columns) and not {
        "cos_vza",
        "cos_sza",
        "cos_raa",
    }.issubset(df.columns):
        df["cos_vza"] = np.cos(np.radians(df["vza"]))
        df["cos_sza"] = np.cos(np.radians(df["sza"]))
        df["cos_raa"] = np.cos(np.radians(df["phi"]))

    # divide all bands starting with B and being numeric by 10000
    band_cols = [col for col in df.columns if re.match(r"^B\d{1,2}[A]?$", col)]
    df[band_cols] = df[band_cols] / 10000.0

    for trait in ["LAI", "FAPAR"]:
        model, upper_lim, band_order = load_grounded_eo_model(variable=trait)
        X = df[band_order].dropna()
        if len(X) == 0:
            continue
        mean, std = model.predict(X, return_std=True)
        mean = np.clip(mean, 0, upper_lim if upper_lim is not None else np.inf)

        out.loc[X.index, f"grounded_{trait.lower()}_mean"] = mean
        out.loc[X.index, f"grounded_{trait.lower()}_std"] = std

    # left join column uuid
    out = out.join(val[["uuid"]])

    return out


def predict_specker() -> pd.DataFrame:
    """Predict LAI with Specker ensemble; returns columns: specker_lai_mean, specker_lai_std."""
    val = load_grounded_eo_validation_data()

    # Band order as used by your main()
    band_order = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "tts",
        "tto",
        "psi",
    ]

    out = pd.DataFrame(index=val.index)
    df = val.copy()

    rename_map = {}
    if "sza" in df.columns:
        rename_map["sza"] = "tts"
    if "vza" in df.columns:
        rename_map["vza"] = "tto"
    if "phi" in df.columns:
        rename_map["phi"] = "psi"

    df = df.rename(columns=rename_map)

    # divide all bands starting with B and being numeric by 10000
    band_cols = [col for col in df.columns if re.match(r"^B\d{1,2}[A]?$", col)]
    df[band_cols] = df[band_cols] / 10000.0

    X = df[band_order].dropna()
    if len(X) == 0:
        return pd.DataFrame(index=val.index)

    for trait in ["laie", "fapar", "fcover"]:
        models = load_specker_model_ensemble(trait=trait)
        preds = np.column_stack([m["pipeline"].predict(X) for m in models.values()])
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)

        out.loc[X.index, f"specker_{trait.lower()}_mean"] = mean
        out.loc[X.index, f"specker_{trait.lower()}_std"] = std

    out = out.join(val[["uuid"]])
    return out


def predict_sl2p() -> pd.DataFrame:
    # Implement prediction logic for SL2P
    val = load_grounded_eo_validation_data()
    out = pd.DataFrame(index=val.index)
    df = val.copy()

    if {"sza", "vza", "phi"}.issubset(df.columns) and not {
        "cosVZA",
        "cosSZA",
        "cosRAA",
    }.issubset(df.columns):
        df["cosVZA"] = np.cos(np.radians(df["vza"]))
        df["cosSZA"] = np.cos(np.radians(df["sza"]))
        df["cosRAA"] = np.cos(np.radians(df["phi"]))

    # 2. pad band numbers (B1 → B01, B2 → B02, … B9 → B09)
    df = df.rename(columns=lambda c: re.sub(r"^B(\d)$", r"B0\1", c))

    # divide all bands starting with B and being numeric by 10000
    band_cols = [col for col in df.columns if re.match(r"^B\d{1,2}[A]?$", col)]
    df[band_cols] = df[band_cols] / 10000.0

    for trait in ["LAI", "fAPAR", "fCOVER"]:
        (
            sl2p_mean_model,
            sl2p_uncertainty_model,
            sl2p_mean_params,
            sl2p_uncertainty_params,
        ) = load_SL2P_model(variable=trait, rel_path="SL2P_PYTHON")
        band_order = sl2p_mean_params["bandorder"]

        X = df[band_order].dropna()

        sl2p_mean = sl2p_mean_model.predict(X, domain_check=False)
        sl2p_std = sl2p_uncertainty_model.predict(X, domain_check=False)

        if trait.lower() == "lai":
            trait_out = "laie"
        else:
            trait_out = trait.lower()
        out.loc[X.index, f"sl2p_{trait_out}_mean"] = sl2p_mean
        out.loc[X.index, f"sl2p_{trait_out}_std"] = sl2p_std

    out = out.join(val[["uuid"]])
    return out


def evaluate_predictions(y_true, y_pred, trait=None, logger=None, groupby=None):
    """
    Evaluate regression predictions with a set of metrics.

    Parameters
    ----------
    y_true : array-like
        Reference or observed values.
    y_pred : array-like
        Model predictions (mean estimates).
    trait : str, optional
        Variable name (e.g., "lai" or "fapar"), used for uncertainty_agreement_ratio().
    logger : logging.Logger, optional
        If provided, metrics will also be logged.
    groupby : array-like, optional
        If provided, metrics will be computed per group and returned for all and per group.

    Returns
    -------
    dict
        Dictionary with computed metrics.
    """

    def _compute_metrics(y_t, y_p, y_t_global=None):
        y_t = np.asarray(y_t).squeeze()
        y_p = np.asarray(y_p).squeeze()

        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p) if len(np.unique(y_t)) > 1 else np.nan
        if y_t_global is not None and len(np.unique(y_t_global)) > 1:
            r2_global = r2_score_oos(y_t, y_p, y_t_global)
        else:
            r2_global = np.nan
        rmse = root_mean_squared_error(y_t, y_p)
        nrmse = rmse / np.mean(y_t) if np.mean(y_t) != 0 else np.nan
        me = np.mean(y_p - y_t)
        n = len(y_t)
        try:
            uar = uncertainty_agreement_ratio(y_t, y_p, variable_name=trait)
        except Exception:
            uar = np.nan

        return {
            "Trait": trait,
            "N": n,
            "MAE": mae,
            "R2": r2,
            "R2_Global": r2_global,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "ME": me,
            "UAR": uar,
        }

    # Build a clean DataFrame to handle NaNs consistently
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if groupby is not None:
        df["group"] = groupby

    # Drop rows with NaNs in required columns
    req_cols = ["y_true", "y_pred"] + (["group"] if groupby is not None else [])
    df = df.dropna(subset=req_cols)

    # No grouping: return dict (back-compat)
    if groupby is None:
        metrics = _compute_metrics(df["y_true"].values, df["y_pred"].values)
        if logger is not None:
            logger.info(f"{trait or 'Trait'} evaluation metrics (ALL):")
            for k, v in metrics.items():
                logger.info(
                    f"  {k}: {v:.4f}" if isinstance(v, (float, int)) else f"  {k}: {v}"
                )
        return metrics

    # Grouped: compute per-group metrics + ALL
    rows = []
    for g, dfg in df.groupby("group"):
        m = _compute_metrics(
            dfg["y_true"].values, dfg["y_pred"].values, df["y_true"].values
        )
        m["Group"] = g
        rows.append(m)

    # Add overall row
    m_all = _compute_metrics(df["y_true"].values, df["y_pred"].values)
    m_all["Group"] = "ALL"
    rows.append(m_all)

    out = pd.DataFrame(rows)[
        ["Group", "Trait", "N", "MAE", "R2", "R2_Global", "RMSE", "NRMSE", "ME", "UAR"]
    ]

    if logger is not None:
        logger.info(f"{trait or 'Trait'} evaluation metrics by group:")
        for _, r in out.iterrows():
            logger.info(
                f"  [{r['Group']}] N={int(r['N'])} MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} "
                f"NRMSE={r['NRMSE']:.4f if pd.notna(r['NRMSE']) else np.nan} R2={r['R2']:.4f if pd.notna(r['R2']) else np.nan} "
                f"ME={r['ME']:.4f} UAR={r['UAR']:.4f if pd.notna(r['UAR']) else np.nan}"
            )

    return out


def build_combined_trait_df(
    validation_data,
    df_sl2p,
    df_specker,
    traits=["fapar", "laie", "fcover"],
    df_grounded=None,
):
    """
    Combine reference and model predictions for all traits (fapar, laie, fcover)
    into a single DataFrame with consistent column naming.
    """

    dfs = []

    for trait in traits:
        # Reference subset
        val_df = validation_data[[trait, f"{trait}_std", "uuid"]].copy()

        sl2p_cols = [c for c in df_sl2p.columns if trait in c.lower()]
        tmp = val_df.merge(df_sl2p[["uuid"] + sl2p_cols], on="uuid", how="left")

        # Merge Specker predictions
        specker_cols = [c for c in df_specker.columns if trait in c.lower()]
        tmp = tmp.merge(df_specker[["uuid"] + specker_cols], on="uuid", how="left")

        # Optionally merge Grounded EO predictions if available
        if df_grounded is not None:
            if trait == "laie":
                trait_temp = "lai"
            else:
                trait_temp = trait
            grounded_cols = [c for c in df_grounded.columns if trait_temp in c.lower()]
            tmp = tmp.merge(
                df_grounded[["uuid"] + grounded_cols], on="uuid", how="left"
            )

        dfs.append(tmp)

    # Merge all traits into a single DataFrame
    df_all_traits = dfs[0]
    for df_other in dfs[1:]:
        df_all_traits = df_all_traits.merge(df_other, on="uuid", how="outer")

    return df_all_traits


def build_summary_table_by_group(
    all_metrics, metrics_order=None, traits_order=None, all_label="ALL"
):
    """
    Make a wide table with rows=(Model, Group) and columns=(Trait, Metric),
    keeping every land-cover group plus the 'ALL' row for each model.

    Parameters
    ----------
    all_metrics : dict
        Nested dict {model: {trait: df}}, where each df has columns:
        ['Group','Trait','N','MAE','R2','R2_Global','RMSE','NRMSE','ME','UAR', ...]
    metrics_order : list[str] or None
        If provided, order metrics in this sequence; defaults to the order found.
    traits_order : list[str] or None
        If provided, order traits in this sequence; defaults to sorted(all traits).
    all_label : str
        Label of the overall row (e.g., 'ALL').

    Returns
    -------
    pd.DataFrame
        MultiIndex columns (Trait, Metric), MultiIndex rows (Model, Group).
    """
    model_frames = []

    # discover all traits if not given
    if traits_order is None:
        traits_order = sorted({t for m in all_metrics.values() for t in m.keys()})

    for model_name, trait_map in all_metrics.items():
        df_model = None

        for trait in traits_order:
            if trait not in trait_map:
                continue
            df_trait = trait_map[trait].copy()

            # keep 'Group' + metric columns (drop 'Trait' if present)
            if "Trait" in df_trait.columns:
                df_trait = df_trait.drop(columns=["Trait"])
            assert (
                "Group" in df_trait.columns
            ), "Each trait DF must contain a 'Group' column."

            metric_cols = [c for c in df_trait.columns if c != "Group"]

            # optionally order metrics
            if metrics_order is not None:
                metric_cols = [c for c in metrics_order if c in metric_cols]

            # rename to MultiIndex (trait, metric)
            cols = ["Group"] + [(trait, c) for c in metric_cols]
            df_trait = df_trait[["Group"] + metric_cols]
            df_trait.columns = cols

            # outer-merge across traits to keep all groups
            if df_model is None:
                df_model = df_trait
            else:
                df_model = df_model.merge(df_trait, on="Group", how="outer")

        # add model id and set composite index
        df_model.insert(0, "Model", model_name)
        df_model = df_model.set_index(["Model", "Group"])
        model_frames.append(df_model)

    # stack models
    summary = pd.concat(model_frames, axis=0).sort_index()

    # make columns a proper MultiIndex with (Trait, Metric)
    # (they already are, but ensure name consistency)
    if not isinstance(summary.columns, pd.MultiIndex):
        # Only 'Group'/'Model' should be index; others are MultiIndex
        pass
    summary.columns = pd.MultiIndex.from_tuples(
        summary.columns, names=["Trait", "Metric"]
    )

    # optional: order columns by traits_order then metrics_order
    if traits_order is not None or metrics_order is not None:
        tuples = list(summary.columns)

        def col_key(tup):
            trait, metric = tup
            tpos = (
                traits_order.index(trait)
                if trait in traits_order
                else len(traits_order)
            )
            mpos = (
                metrics_order.index(metric)
                if (metrics_order and metric in metrics_order)
                else 999
            )
            return (tpos, mpos, metric)

        tuples_sorted = sorted(tuples, key=col_key)
        summary = summary[tuples_sorted]

    # nice sort of groups: keep ALL last within each model
    # reindex the row MultiIndex to place ALL at bottom per model
    def sort_groups(idx):
        # idx is a MultiIndex (Model, Group)
        df_idx = idx.to_frame(index=False)
        df_idx["_grp_order"] = (df_idx["Group"] == all_label).astype(
            int
        )  # 0 for normal, 1 for ALL
        order = df_idx.sort_values(["Model", "_grp_order", "Group"]).index
        return idx[order]

    summary = summary.reindex(sort_groups(summary.index))

    return summary


def create_revisions_table1():
    # load validation data
    validation_data = load_grounded_eo_validation_data()

    # df_grounded = predict_grounded_eo()
    df_specker = predict_specker()
    df_sl2p = predict_sl2p()

    df_all_traits = build_combined_trait_df(validation_data, df_sl2p, df_specker)

    # join back metadata from validation data
    df_all_traits = df_all_traits.merge(
        validation_data[["uuid", "Site", "ECO_ID", "NLCD", "Network", "Plot"]],
        on="uuid",
        how="left",
    )

    all_metrics = {}

    models = ["sl2p", "specker"]  # , "grounded"]
    all_metrics = {}
    for model in models:
        all_metrics[model] = {}
    for trait in ["fapar", "laie", "fcover"]:
        for model in models:
            y_true = df_all_traits[trait]

            y_pred = df_all_traits[f"{model}_{trait}_mean"]
            metrics = evaluate_predictions(
                y_true, y_pred, trait=trait, groupby=df_all_traits["NLCD"]
            )
            all_metrics[model][trait] = metrics
            print(f"{model.upper()} - {trait}:")
            print(metrics)
            print()

    #
    metrics_order = ["N", "MAE", "R2", "R2_Global", "RMSE", "NRMSE", "ME", "UAR"]

    summary = build_summary_table_by_group(
        all_metrics,
        metrics_order=metrics_order,
        traits_order=["laie", "fapar", "fcover"],  # or None for automatic
        all_label="ALL",
    )

    summary = summary.sort_index(level=[0, 1], ascending=[False, True])
    # cast N to int: remove second and third N columns if present
    n_cols = [col for col in summary.columns if col[1] == "N"]
    if len(n_cols) > 0:
        first_n_col = n_cols[0]
        summary[first_n_col] = summary[first_n_col].astype(int)
        for col in n_cols[1:]:
            summary = summary.drop(columns=[col])

    # round MAE, R2, R2_Global, RMSE, NRMSE, ME, UAR to 2 decimals
    for col in summary.columns:
        if col[1] in ["RMSE", "MAE", "R2", "R2_Global", "NRMSE", "ME", "UAR"]:
            summary[col] = summary[col].round(2)

    print("Summary Table:")
    print(summary)
    # save as csv and open in excel for final formatting
    summary.to_csv("tables/revision_table1_model_comparison.csv", float_format="%.2f")
    print("Saved summary table to revision_table1_model_comparison.csv")

    # save to latex directly:
    summary.to_latex(
        "tables/revision_table1_model_comparison.tex",
        float_format="%.2f",
        multirow=True,
    )
    print("Saved summary table to revision_table1_model_comparison.tex")


def create_revisions_supplementary_table_XX():
    # load validation data
    validation_data = load_grounded_eo_validation_data()

    # df_grounded = predict_grounded_eo()
    df_specker = predict_specker()
    df_sl2p = predict_sl2p()
    df_grounded = predict_grounded_eo()

    df_all_traits = build_combined_trait_df(
        validation_data, df_sl2p, df_specker, df_grounded=df_grounded
    )

    # join back metadata from validation data
    df_all_traits = df_all_traits.merge(
        validation_data[["uuid", "Site", "ECO_ID", "NLCD", "Network", "Plot"]],
        on="uuid",
        how="left",
    )

    all_metrics = {}

    models = ["sl2p", "specker", "grounded"]
    all_metrics = {}
    for model in models:
        all_metrics[model] = {}
    for trait in ["fapar"]:
        for model in models:
            y_true = df_all_traits[trait]

            if model == "grounded" and trait == "laie":
                y_pred = df_all_traits["grounded_lai_mean"]
            else:
                y_pred = df_all_traits[f"{model}_{trait}_mean"]
            metrics = evaluate_predictions(
                y_true, y_pred, trait=trait, groupby=df_all_traits["NLCD"]
            )
            all_metrics[model][trait] = metrics
            print(f"{model.upper()} - {trait}:")
            print(metrics)
            print()

    #
    metrics_order = ["N", "MAE", "R2", "R2_Global", "RMSE", "NRMSE", "ME", "UAR"]

    summary = build_summary_table_by_group(
        all_metrics,
        metrics_order=metrics_order,
        traits_order=["fapar"],
        all_label="ALL",
    )

    summary = summary.sort_index(level=[0, 1], ascending=[False, True])

    # cast N to int: remove second and third N columns if present
    n_cols = [col for col in summary.columns if col[1] == "N"]
    if len(n_cols) > 0:
        first_n_col = n_cols[0]
        summary[first_n_col] = summary[first_n_col].astype(int)
        for col in n_cols[1:]:
            summary = summary.drop(columns=[col])

    # round MAE, R2, R2_Global, RMSE, NRMSE, ME, UAR to 2 decimals
    for col in summary.columns:
        if col[1] in ["RMSE", "MAE", "R2", "R2_Global", "NRMSE", "ME", "UAR"]:
            summary[col] = summary[col].round(2)

    print("Summary Table:")
    print(summary)
    # save as csv and open in excel for final formatting
    summary.to_csv(
        "tables/revision_supplementary_table_model_comparison.csv", float_format="%.2f"
    )
    print("Saved summary table to revision_supplementary_table_model_comparison.csv")

    # save to latex directly:
    summary.to_latex(
        "tables/revision_supplementary_table_model_comparison.tex",
        float_format="%.2f",
        multirow=True,
    )
    print("Saved summary table to revision_supplementary_table_model_comparison.tex")

    # val_fapar = validation_data[["fapar", "fapar_std", "uuid"]]

    # for calibration test: use fapar, and lai

    # # compare all models
    # results = []
    # results.append(calculate_metrics(y_val, specker_mean, "Specker"))
    # results.append(calculate_metrics(y_val, sl2p_mean, "SL2P"))
    # results.append(calculate_metrics(y_val, grounded_mean, "Grounded EO"))
    # # put results in dataframe
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("model_comparison_lai.csv", index=False)
    # logger.info("Saved results to model_comparison_lai.csv")
    # print(results_df)


def create_site_table_supplement():
    pass
    validation_data = load_grounded_eo_validation_data()

    cols = ["uuid", "Site", "ECO_ID", "NLCD", "Network", "Plot"]

    site_table = validation_data[cols].drop_duplicates().reset_index(drop=True)

    # count N per site
    site_counts = (
        validation_data.groupby("Site")["uuid"]
        .count()
        .reset_index()
        .rename(columns={"uuid": "N"})
    )

    # count plots per site
    plot_counts = (
        validation_data.groupby("Site")["Plot"]
        .nunique()
        .reset_index()
        .rename(columns={"Plot": "N_Plots"})
    )

    modal_nlcd = (
        validation_data.groupby(["Site", "NLCD"])["uuid"]
        .count()
        .reset_index()
        .sort_values(["Site", "uuid"], ascending=[True, False])
        .drop_duplicates(subset=["Site"], keep="first")
        .rename(columns={"NLCD": "Modal_NLCD"})
    )

    model_ecoid = (
        validation_data.groupby(["Site", "ECO_ID"])["uuid"]
        .count()
        .reset_index()
        .sort_values(["Site", "uuid"], ascending=[True, False])
        .drop_duplicates(subset=["Site"], keep="first")
        .rename(columns={"ECO_ID": "Modal_ECO_ID"})
    )

    site_table = site_table.merge(site_counts, on="Site", how="left")
    site_table = site_table.merge(plot_counts, on="Site", how="left")
    site_table = site_table.merge(
        modal_nlcd[["Site", "Modal_NLCD"]], on="Site", how="left"
    )
    site_table = site_table.merge(
        model_ecoid[["Site", "Modal_ECO_ID"]], on="Site", how="left"
    )

    # drop uuid, drop duplicates again
    site_table = (
        site_table.drop(columns=["uuid", "Plot", "NLCD", "ECO_ID"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # rename Modal_ECO_ID to ECO_ID, Modal_NLCD to NLCD
    site_table = site_table.rename(
        columns={"Modal_ECO_ID": "ECO_ID", "Modal_NLCD": "NLCD"}
    )

    # load biome ecoregion lookup table and add Biome column
    ecoregion_lookup = pd.read_csv(
        "data/misc/ecoregion_biome_table.csv", usecols=["ECO_ID", "BIOME_NUM"]
    )

    site_table = site_table.merge(ecoregion_lookup, on="ECO_ID", how="left")

    # colname BIOME_NUM to Biome  and to int
    site_table = site_table.rename(columns={"BIOME_NUM": "Biome"})
    site_table["Biome"] = site_table["Biome"].astype(pd.Int64Dtype())

    #

    print("Site Table Supplement:")
    print(site_table)


def create_predictions_figure_1():
    # load validation data
    validation_data = load_grounded_eo_validation_data()

    # df_grounded = predict_grounded_eo()
    df_specker = predict_specker()
    df_sl2p = predict_sl2p()

    df_all_traits = build_combined_trait_df(validation_data, df_sl2p, df_specker)

    # join back metadata from validation data
    df_all_traits = df_all_traits.merge(
        validation_data[["uuid", "Site", "ECO_ID", "NLCD", "Network", "Plot"]],
        on="uuid",
        how="left",
    )

    # save
    df_all_traits.to_csv(
        "data/train_pipeline/output/predictions_specker_sl2p.csv", index=False
    )


if __name__ == "__main__":
    create_predictions_figure_1()
    # create_revisions_table1()
    # create_revisions_supplementary_table_XX()
    # create_site_table_supplement()
    # main()
    # main()
    # main()
    # main()
    # main()
