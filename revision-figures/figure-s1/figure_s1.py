import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from train_pipeline.predict_insitu_comparison import (
    build_combined_trait_df,
    predict_grounded_eo,
    predict_sl2p,
    predict_specker,
)
from train_pipeline.utils_loading import load_grounded_eo_validation_data

SMOOTH_KNOTS_ATTACHEMENT = {
    "laie": {"min": -0.5, "max": 5.5},
    "fapar": {"min": -0.1, "max": 1.1},
    "fcover": {"min": -0.1, "max": 1.1},
}


def compute_calibration_metrics(y_true, y_pred, sigma_pred):
    """
    Computes:
    - MACE: Mean Absolute Calibration Error
    - RMCE: Root Mean Calibration Error
    """

    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    mace = np.mean(np.abs(abs_errors - sigma_pred))
    rmce = np.sqrt(np.mean((abs_errors - sigma_pred) ** 2))

    return mace, rmce


def calibrate_sigma_mace(y_true, y_pred, sigma_pred):
    """Find optimal tau to scale sigma_pred by minimizing MACE."""

    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    def mace_tau(tau):
        return np.mean(np.abs(abs_errors - tau * sigma_pred))

    opt = minimize(
        lambda t: mace_tau(t[0]),
        x0=np.array([1.0]),
        bounds=[(1e-6, 100)],
        method="L-BFGS-B",
    )
    return opt.x[0]


def smooth_tau_calibration(
    y_true,
    y_pred,
    sigma_pred,
    n_knots,
    spline_degree,
    trait=Literal["laie", "fapar", "fcover"],
):
    """
    Learn a smooth tau(y_pred) via a cubic spline.

    Parameters
    ----------
    y_true : array-like
        Ground truth targets.
    y_pred : array-like
        Predicted means.
    sigma_pred : array-like
        Predicted stds (uncalibrated).
    n_knots : int
        Number of internal knots for spline (controls smoothness).
        Typical: 6 (recommended).
    spline_degree : int
        Degree of spline (3 recommended)

    trait: str

    Returns
    -------
    tau_func : callable
        Function tau(y_pred) returning the scaling factor.
    sigma_cal : array
        Calibrated stds = tau(y_pred) * sigma_pred.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma_pred = np.asarray(sigma_pred)

    abs_err = np.abs(y_true - y_pred)

    # Sort by prediction for stable spline fitting
    idx = np.argsort(y_pred)
    yp = y_pred[idx]
    abs_err_sorted = abs_err[idx]
    sigma_sorted = sigma_pred[idx]

    # Fix the spline to tau = 1 outside the domain:
    knots = np.linspace(
        SMOOTH_KNOTS_ATTACHEMENT[trait]["min"],
        SMOOTH_KNOTS_ATTACHEMENT[trait]["max"],
        n_knots,
    )

    # Need to add boundary knots repeated degree times
    t = np.concatenate(
        (np.repeat(knots[0], spline_degree), knots, np.repeat(knots[-1], spline_degree))
    )

    # Initialize spline coefficients (all 1 => initial tau=1)
    c0 = np.ones(len(t) - spline_degree - 1)

    # Build basis matrix for speed
    basis = np.vstack(
        [
            BSpline(t, (np.arange(len(c0)) == j).astype(float), spline_degree)(yp)
            for j in range(len(c0))
        ]
    ).T

    # Objective: MACE with smooth tau
    def objective(c):
        tau_vals = basis.dot(c)
        tau_vals = np.clip(tau_vals, 1e-6, 100)  # keep τ positive and sane
        sigma_cal = tau_vals * sigma_sorted
        return np.mean(np.abs(abs_err_sorted - sigma_cal))

    # Optimize spline coefficients
    res = minimize(objective, c0, method="L-BFGS-B", bounds=[(1e-6, 100)] * len(c0))
    c_opt = res.x

    # Build final spline
    tau_spline = BSpline(t, c_opt, spline_degree)

    # Calibrated sigma
    sigma_cal = tau_spline(y_pred) * sigma_pred

    return tau_spline, sigma_cal


def plot_uncertainty_intervals(
    y_true,
    y_pred,
    std_unc,
    std_cal,
    N=100,
    ylab="Value",
    title="Uncertainty Calibration",
    save_path: Optional[str] = None,
):
    """
    Plot sorted predictions with uncalibrated and calibrated 95% intervals.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted mean values.
    std_unc : array-like
        Uncalibrated predictive standard deviations.
    std_cal : array-like
        Calibrated predictive standard deviations.
    N : int
        Number of samples to subsample for plotting.
    ylab : str
        Y-axis label.
    title : str
        Plot title.
    """

    # ---- Convert to arrays ----
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    std_unc = np.asarray(std_unc)
    std_cal = np.asarray(std_cal)

    # ---- Sort by predicted values ----
    idx = np.argsort(y_pred)
    y_true_s = y_true[idx]
    y_pred_s = y_pred[idx]
    std_unc_s = std_unc[idx]
    std_cal_s = std_cal[idx]

    # ---- Subsample evenly ----
    if len(y_pred_s) > N:
        sel = np.linspace(0, len(y_pred_s) - 1, N).astype(int)
        y_true_s = y_true_s[sel]
        y_pred_s = y_pred_s[sel]
        std_unc_s = std_unc_s[sel]
        std_cal_s = std_cal_s[sel]

    # ---- 95% intervals ----
    uncal_lower = y_pred_s - 1.96 * std_unc_s
    uncal_upper = y_pred_s + 1.96 * std_unc_s

    cal_lower = y_pred_s - 1.96 * std_cal_s
    cal_upper = y_pred_s + 1.96 * std_cal_s

    # ---- Plot ----
    plt.figure(figsize=(12, 6))

    # Uncalibrated band
    plt.fill_between(
        range(len(y_pred_s)),
        uncal_lower,
        uncal_upper,
        color="blue",
        alpha=0.20,
        label="Uncalibrated 95% interval",
    )

    # Calibrated band
    plt.fill_between(
        range(len(y_pred_s)),
        cal_lower,
        cal_upper,
        color="orange",
        alpha=0.20,
        label="Calibrated 95% interval",
    )

    # Prediction line
    plt.plot(y_pred_s, color="black", lw=2, label="Predicted")

    # True values
    plt.scatter(
        range(len(y_true_s)), y_true_s, color="red", s=18, label="True", alpha=0.85
    )

    plt.xlabel("Sorted samples")
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def calibrate_constant_tau():
    for trait in ["fcover", "fapar", "laie"]:
        # ---- Load data ----
        validation_data = load_grounded_eo_validation_data()
        df_specker = predict_specker(df=validation_data)
        df_sl2p = predict_sl2p(df=validation_data)
        df_grounded = predict_grounded_eo(df=validation_data)

        df_all = build_combined_trait_df(
            validation_data, df_sl2p, df_specker, df_grounded=df_grounded
        )

        cols = [
            trait,
            f"{trait}_std",
            f"specker_{trait}_mean",
            f"specker_{trait}_std",
            f"sl2p_{trait}_mean",
            f"sl2p_{trait}_std",
        ]
        if trait == "fapar":
            cols += [
                f"grounded_{trait}_mean",
                f"grounded_{trait}_std",
            ]

        # rename insitu columns
        df = df_all[cols].rename(
            columns={
                trait: f"insitu_{trait}_mean",
                f"{trait}_std": f"insitu_{trait}_std",
            }
        )

        # ===============================================================
        # Compare SPECKER vs SL2P uncertainty
        # ===============================================================

        results = []

        for model in ["sl2p", "specker"]:
            y_pred = df[f"{model}_{trait}_mean"].values
            sigma_pred = df[f"{model}_{trait}_std"].values

            # ---- Compute MACE & RMCE ----
            mace, rmce = compute_calibration_metrics(
                df[f"insitu_{trait}_mean"].values, y_pred, sigma_pred
            )

            # ---- Calibrate τ via MACE minimization ----
            tau_opt = calibrate_sigma_mace(
                df[f"insitu_{trait}_mean"].values, y_pred, sigma_pred
            )

            results.append(dict(model=model.upper(), MACE=mace, RMCE=rmce, tau=tau_opt))

            df[f"{model}_{trait}_calibrated_std"] = tau_opt * sigma_pred

            # add calibrated calibration scores
            mace_calibrated, rmce_calibrated = compute_calibration_metrics(
                df[f"insitu_{trait}_mean"].values,
                y_pred,
                df[f"{model}_{trait}_calibrated_std"].values,
            )

            results.append(
                dict(
                    model=f"{model.upper()}_calibrated",
                    MACE=mace_calibrated,
                    RMCE=rmce_calibrated,
                    tau=tau_opt,
                )
            )

        # display table
        df_results = pd.DataFrame(results)
        print("\n=== Uncertainty Comparison (SPECKER vs SL2P) ===")
        print(df_results)
        print()

        # ===============================================================
        # Visualize calibrated vs uncalibrated intervals for SL2P
        # ===============================================================

        model = "specker"
        plot_uncertainty_intervals(
            y_true=df[f"insitu_{trait}_mean"].values,
            y_pred=df[f"{model}_{trait}_mean"].values,
            std_unc=df[f"{model}_{trait}_std"].values,
            std_cal=df[f"{model}_{trait}_calibrated_std"].values,
            N=100,
            ylab=trait,
            title=f"{model.upper()} - Uncertainty Calibration Comparison",
            save_path=f"revision-figures/figure-s1/{model}_{trait}_linear_tau",
        )

        model = "sl2p"
        plot_uncertainty_intervals(
            y_true=df[f"insitu_{trait}_mean"].values,
            y_pred=df[f"{model}_{trait}_mean"].values,
            std_unc=df[f"{model}_{trait}_std"].values,
            std_cal=df[f"{model}_{trait}_calibrated_std"].values,
            N=100,
            ylab=trait,
            title=f"{model.upper()} - Uncertainty Calibration Comparison",
            save_path=f"revision-figures/figure-s1/{model}_{trait}_linear_tau",
        )

    # plot coefficients /


def calibrate_smooth_tau(return_df: bool = False):
    if return_df:
        dfs = {}

    for trait in ["fcover", "fapar", "laie"]:
        # ---- Load data ----
        validation_data = load_grounded_eo_validation_data()
        df_specker = predict_specker(df=validation_data)
        df_sl2p = predict_sl2p(df=validation_data)
        df_grounded = predict_grounded_eo(df=validation_data)

        df_all = build_combined_trait_df(
            validation_data, df_sl2p, df_specker, df_grounded=df_grounded
        )

        # rename insitu columns
        cols = [
            trait,
            f"{trait}_std",
            f"specker_{trait}_mean",
            f"specker_{trait}_std",
            f"sl2p_{trait}_mean",
            f"sl2p_{trait}_std",
        ]
        if trait == "fapar":
            cols += [
                f"grounded_{trait}_mean",
                f"grounded_{trait}_std",
            ]

        # rename insitu columns
        df = df_all[cols].rename(
            columns={
                trait: f"insitu_{trait}_mean",
                f"{trait}_std": f"insitu_{trait}_std",
            }
        )

        # ===============================================================
        # Compare SPECKER vs SL2P uncertainty
        # ===============================================================

        results = []

        for model in ["sl2p", "specker", "grounded"]:
            if trait != "fapar" and model == "grounded":
                continue
            y_pred = df[f"{model}_{trait}_mean"].values
            sigma_pred = df[f"{model}_{trait}_std"].values

            # ---- Compute MACE & RMCE ----
            mace, rmce = compute_calibration_metrics(
                df[f"insitu_{trait}_mean"].values, y_pred, sigma_pred
            )

            # ---- Calibrate τ via MACE minimization ----
            tau_opt, sigma_cal_smooth = smooth_tau_calibration(
                y_true=df[f"insitu_{trait}_mean"].values,
                y_pred=df[f"{model}_{trait}_mean"].values,
                sigma_pred=df[f"{model}_{trait}_std"].values,
                n_knots=6,
                spline_degree=3,
                trait=trait,
            )

            results.append(dict(model=model.upper(), MACE=mace, RMCE=rmce, tau=tau_opt))

            df[f"{model}_{trait}_calibrated_std"] = sigma_cal_smooth

            # add calibrated calibration scores
            mace_calibrated, rmce_calibrated = compute_calibration_metrics(
                df[f"insitu_{trait}_mean"].values,
                y_pred,
                df[f"{model}_{trait}_calibrated_std"].values,
            )

            results.append(
                dict(
                    model=f"{model.upper()}_calibrated",
                    MACE=mace_calibrated,
                    RMCE=rmce_calibrated,
                )
            )

        # display table
        df_results = pd.DataFrame(results)
        print("\n=== Uncertainty Comparison (SPECKER vs SL2P) ===")
        print(df_results)
        print()

        # save results as csv
        os.makedirs("posthoc-calibration/results", exist_ok=True)
        df_results.to_csv(
            f"posthoc-calibration/results/{model}_{trait}_smooth_tau_calibration_results.csv",
            index=False,
        )

        # ===============================================================
        # Visualize calibrated vs uncalibrated intervals for SL2P
        # ===============================================================

        model = "specker"
        plot_uncertainty_intervals(
            y_true=df[f"insitu_{trait}_mean"].values,
            y_pred=df[f"{model}_{trait}_mean"].values,
            std_unc=df[f"{model}_{trait}_std"].values,
            std_cal=df[f"{model}_{trait}_calibrated_std"].values,
            N=100,
            ylab=trait,
            title=f"{model.upper()} - Uncertainty Calibration Comparison",
            save_path=f"revision-figures/figure-s1/subplots/{model}_{trait}_smooth_tau",
        )

        model = "sl2p"
        plot_uncertainty_intervals(
            y_true=df[f"insitu_{trait}_mean"].values,
            y_pred=df[f"{model}_{trait}_mean"].values,
            std_unc=df[f"{model}_{trait}_std"].values,
            std_cal=df[f"{model}_{trait}_calibrated_std"].values,
            N=100,
            ylab=trait,
            title=f"{model.upper()} - Uncertainty Calibration Comparison",
            save_path=f"revision-figures/figure-s1/subplots/{model}_{trait}_smooth_tau",
        )

        if trait == "fapar":
            model = "grounded"
            plot_uncertainty_intervals(
                y_true=df[f"insitu_{trait}_mean"].values,
                y_pred=df[f"{model}_{trait}_mean"].values,
                std_unc=df[f"{model}_{trait}_std"].values,
                std_cal=df[f"{model}_{trait}_calibrated_std"].values,
                N=100,
                ylab=trait,
                title=f"{model.upper()} - Uncertainty Calibration Comparison",
                save_path=f"revision-figures/figure-s1/subplots/{model}_{trait}_smooth_tau",
            )

        if return_df:
            dfs[trait] = df

    if return_df:
        return dfs
    else:
        return None


def figure_smooth_tau(dfs: dict):
    # Supplementary figure S5:
    # dfs contains the dataframes returned by calibrate_smooth_tau(return_df=True): dfs[trait] --> dataframe

    # 2 columns (model s2biophys and sl2p), 3 rows (traits): plot_uncertainty_intervals for each
    #   - joint y axis limits per trait
    #   - add legend only to first plot
    #   - title of first row contains model name
    #   - one join legend: predicted, in-situ RM, uncalibrated 95% interval, calibrated 95% interval

    traits = ["laie", "fcover", "fapar"]
    # models = ["specker", "sl2p", "grounded"]
    models = ["specker", "sl2p"]

    N = 100  # number of points to plot

    # Create multi-panel figure
    if "grounded" in models:
        fig, axes = plt.subplots(
            nrows=3, ncols=3, figsize=(14, 14), sharex=False, sharey="row"
        )
    else:
        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(14, 12), sharex=False, sharey="row"
        )

    col_unc = "blue"
    col_cal = "orange"
    col_pred = "black"
    col_true = "red"
    col_tau = "darkgreen"

    for i, trait in enumerate(traits):
        df = dfs[trait]
        y_true_full = df[f"insitu_{trait}_mean"].values

        for j, model in enumerate(models):
            if trait != "fapar" and model == "grounded":
                continue
            row_label = i + 1
            if j == 0:
                sublabel = f"A{row_label}"
            else:
                sublabel = f"B{row_label}"

            ax = axes[i, j]
            ax.text(
                0.015,
                0.97,
                sublabel,
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top",
                ha="left",
            )

            # --- Extract relevant arrays ---
            y_pred = df[f"{model}_{trait}_mean"].values
            std_unc = df[f"{model}_{trait}_std"].values
            std_cal = df[f"{model}_{trait}_calibrated_std"].values

            # --- Sort by prediction ---
            idx = np.argsort(y_pred)
            y_true = y_true_full[idx]
            y_pred_s = y_pred[idx]
            std_unc_s = std_unc[idx]
            std_cal_s = std_cal[idx]

            # --- Subsample evenly ---
            if len(y_pred_s) > N:
                sel = np.linspace(0, len(y_pred_s) - 1, N + 2).astype(int)[1:-1]
                y_true = y_true[sel]
                y_pred_s = y_pred_s[sel]
                std_unc_s = std_unc_s[sel]
                std_cal_s = std_cal_s[sel]

            # ---- 95% intervals ----
            uncal_lower = y_pred_s - 1.96 * std_unc_s
            uncal_upper = y_pred_s + 1.96 * std_unc_s
            cal_lower = y_pred_s - 1.96 * std_cal_s
            cal_upper = y_pred_s + 1.96 * std_cal_s

            x = np.arange(len(y_pred_s))

            # ---- Plot intervals ----
            ax.fill_between(x, uncal_lower, uncal_upper, color=col_unc, alpha=0.20)
            ax.fill_between(x, cal_lower, cal_upper, color=col_cal, alpha=0.20)

            # Prediction line
            ax.plot(x, y_pred_s, color=col_pred, lw=2)

            # True values
            ax.scatter(x, y_true, s=18, color=col_true, alpha=0.85)

            # Row titles (y-axis labels)
            y_labels = {
                "fcover": "FCOVER",
                "fapar": "FAPAR",
                "laie": "LAIe [m²/m²]",
            }
            if j == 0:
                ax.set_ylabel(y_labels[trait], fontsize=12)

            # Column titles only for top row
            if i == 0:
                model_names = {"specker": "S2BIOPHYS", "sl2p": "SL2P"}
                ax.set_title(model_names[model], fontsize=14)

            if i == 2:
                ax.set_xlabel("Sorted samples", fontsize=12)

            if trait == "fapar" or trait == "fcover":
                ax.set_ylim(-0.1, 1.1)
                # add dashed line at y=0 and y=1
                ax.axhline(0, color="gray", linestyle="--", lw=0.8)
                ax.axhline(1, color="gray", linestyle="--", lw=0.8)
            elif trait == "laie":
                ax.set_ylim(-0.5, 5.5)
                # add dashed line at y=0
                ax.axhline(0, color="gray", linestyle="--", lw=0.8)
            else:
                raise ValueError(f"Unknown trait: {trait}")

                # τ curve
            tau_s = std_cal_s / std_unc_s
            ax2 = ax.twinx()
            ax2.plot(x, tau_s, color=col_tau, lw=1.5)
            ax2.set_ylabel("τ", color=col_tau)
            ax2.tick_params(axis="y", colors=col_tau)
            ax2.set_zorder(1)  # keep tau-axis behind main axis
            ax.patch.set_visible(False)

            # --- Compute MACE ---
            mace_unc, _ = compute_calibration_metrics(y_true_full, y_pred, std_unc)
            mace_cal, _ = compute_calibration_metrics(y_true_full, y_pred, std_cal)

            # # --- Add MACE text inside each subplot ---
            # ax.text(
            #     0.02,
            #     0.92,
            #     f"MACE uncalibrated: \n{mace_unc:.3f}\nMACE calibrated: \n{mace_cal:.3f}",
            #     transform=ax.transAxes,
            #     ha="left",
            #     va="top",
            #     fontsize=10,
            #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            # )

            # ax.text(
            #     0.02,
            #     0.96,
            #     f"MACE unc/cal: {mace_unc:.3f} / {mace_cal:.3f}",
            #     transform=ax.transAxes,
            #     ha="left",
            #     va="top",
            #     fontsize=11,
            # )

            ax.text(
                0.02,
                0.88,
                "MACE:",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
            )
            ax.text(
                0.02,
                0.81,
                f"- unc: {mace_unc:.3f}",
                color=col_unc,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
            )
            ax.text(
                0.02,
                0.74,
                f"- cal:  {mace_cal:.3f}",
                color=col_cal,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
            )

    # ---- Shared Legend ----
    handles = [
        plt.Line2D([], [], color=col_pred, lw=2, label="Predicted"),
        plt.Line2D(
            [],
            [],
            color=col_true,
            marker="o",
            linestyle="None",
            markersize=6,
            label="In-situ RM",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=col_unc, alpha=0.2, label="Uncalibrated 95% interval"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=col_cal, alpha=0.2, label="Calibrated 95% interval"
        ),
        # plt.Line2D([], [], color=col_tau, lw=2, label="τ(ŷ)"),  # <-- NEW
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    print("Saving figure_smooth_tau...")
    plt.savefig("revision-figures/figure-s1/figure_s1.png", dpi=300)


if __name__ == "__main__":
    # calibrate_constant_tau()
    dfs = calibrate_smooth_tau(return_df=True)

    figure_smooth_tau(dfs)
