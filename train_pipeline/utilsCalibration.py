from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import minimize

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
    recalibration_table : pd.DataFrame
        DataFrame with columns: y_pred, tau(y_pred) with y_pred ranging from SMOOTH_KNOTS_ATTACHEMENT[trait]["min"] to SMOOTH_KNOTS_ATTACHEMENT[trait]["max"]
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

    # Build recalibration table
    y_pred_grid = np.linspace(
        SMOOTH_KNOTS_ATTACHEMENT[trait]["min"],
        SMOOTH_KNOTS_ATTACHEMENT[trait]["max"],
        100,
    )
    tau_values = tau_spline(y_pred_grid)

    recalibration_table = pd.DataFrame({"y_pred": y_pred_grid, "tau": tau_values})

    return tau_spline, sigma_cal, recalibration_table


# if __name__ == "__main__":
# calibrate_smooth_tau(variable="laie")
