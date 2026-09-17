#!/usr/bin/env python3
"""Summarize and plot paired LAIe heterogeneity differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = {
    "mean": ("Mean LAI$_e$", "higher"),
    "sd": ("LAI$_e$ standard deviation", "lower"),
    # "rao_q": ("Rao's Q", "lower"),
    "adjacent_difference": ("Adjacent-pixel difference", "lower"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument(
        "--control-type",
        choices=("naturally_regenerating", "primary", "all"),
        default="naturally_regenerating",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bootstrap_statistics(
    differences: np.ndarray,
    *,
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for mean, median, Cohen's dz and direction."""
    n = differences.size
    means = np.empty(samples)
    medians = np.empty(samples)
    effect_sizes = np.empty(samples)
    direction = np.empty(samples)

    # Chunking avoids allocating the complete bootstrap index matrix.
    chunk_size = 2_000
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        boot = differences[rng.integers(0, n, size=(stop - start, n))]
        boot_means = boot.mean(axis=1)
        boot_sd = boot.std(axis=1, ddof=1)
        means[start:stop] = boot_means
        medians[start:stop] = np.median(boot, axis=1)
        effect_sizes[start:stop] = np.divide(
            boot_means,
            boot_sd,
            out=np.full_like(boot_means, np.nan),
            where=boot_sd > 0,
        )
        direction[start:stop] = (boot < 0).mean(axis=1)

    alpha = 1 - confidence_level
    quantiles = [alpha / 2, 1 - alpha / 2]
    return {
        "mean": tuple(np.quantile(means, quantiles)),
        "median": tuple(np.quantile(medians, quantiles)),
        "cohen_dz": tuple(np.nanquantile(effect_sizes, quantiles)),
        "proportion_lower": tuple(np.quantile(direction, quantiles)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1_000:
        raise ValueError("Use at least 1,000 bootstrap samples.")
    if not 0 < args.confidence_level < 1:
        raise ValueError("confidence-level must lie between 0 and 1.")

    data = pd.read_csv(args.input)
    if "passes_valid_fraction" in data:
        valid_flag = data["passes_valid_fraction"]
        if valid_flag.dtype != bool:
            valid_flag = valid_flag.astype(str).str.lower().eq("true")
        data = data.loc[valid_flag].copy()
    if args.control_type != "all":
        data = data.loc[data["control_type"] == args.control_type].copy()
    if data.empty:
        raise ValueError("No pairs remain after filtering.")

    rng = np.random.default_rng(args.seed)
    records = []
    distributions: dict[str, np.ndarray] = {}

    for metric, (label, expected_direction) in METRICS.items():
        column = f"difference_{metric}"
        differences = data[column].dropna().to_numpy(dtype=float)
        if differences.size < 2:
            raise ValueError(f"Too few valid observations in {column}.")
        distributions[metric] = differences

        intervals = bootstrap_statistics(
            differences,
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            rng=rng,
        )
        sd_difference = differences.std(ddof=1)
        statistic, p_value = wilcoxon(
            differences, alternative="two-sided", zero_method="wilcox"
        )
        proportion_lower = float(np.mean(differences < 0))
        proportion_expected = (
            1 - proportion_lower if expected_direction == "higher" else proportion_lower
        )
        if expected_direction == "higher":
            expected_ci = (
                1 - intervals["proportion_lower"][1],
                1 - intervals["proportion_lower"][0],
            )
        else:
            expected_ci = intervals["proportion_lower"]
        records.append(
            {
                "metric": metric,
                "label": label.replace("$", ""),
                "n_pairs": differences.size,
                "mean_difference": differences.mean(),
                "mean_ci_lower": intervals["mean"][0],
                "mean_ci_upper": intervals["mean"][1],
                "median_difference": np.median(differences),
                "median_ci_lower": intervals["median"][0],
                "median_ci_upper": intervals["median"][1],
                "sd_difference": sd_difference,
                "cohen_dz": differences.mean() / sd_difference,
                "cohen_dz_ci_lower": intervals["cohen_dz"][0],
                "cohen_dz_ci_upper": intervals["cohen_dz"][1],
                "expected_direction": expected_direction,
                "proportion_expected_direction": proportion_expected,
                "proportion_expected_ci_lower": expected_ci[0],
                "proportion_expected_ci_upper": expected_ci[1],
                "proportion_lower": proportion_lower,
                "proportion_lower_ci_lower": intervals["proportion_lower"][0],
                "proportion_lower_ci_upper": intervals["proportion_lower"][1],
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
        )

    summary = pd.DataFrame(records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
    for ax, record in zip(axes.flat, records, strict=True):
        values = distributions[record["metric"]]
        ax.hist(
            values,
            bins="fd",
            density=True,
            color="#4C78A8",
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvspan(
            record["mean_ci_lower"],
            record["mean_ci_upper"],
            color="#E45756",
            alpha=0.18,
            label="95% CI of mean",
        )
        ax.axvline(
            record["mean_difference"],
            color="#E45756",
            linewidth=2,
            label="Mean difference",
        )
        ax.set_title(record["label"])
        ax.set_xlabel("Plantation − control")
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.03,
            0.95,
            f"mean = {record['mean_difference']:.3f}\n"
            f"95% CI [{record['mean_ci_lower']:.3f}, "
            f"{record['mean_ci_upper']:.3f}]\n"
            f"$d_z$ = {record['cohen_dz']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"Paired LAI$_e$ differences: plantation versus "
        f"{args.control_type.replace('_', ' ')} (n = {len(data)})",
        fontsize=13,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved summary to {args.summary}")
    print(f"Saved figure to {args.figure}")
    print(
        summary[
            [
                "metric",
                "n_pairs",
                "mean_difference",
                "mean_ci_lower",
                "mean_ci_upper",
                "cohen_dz",
                "cohen_dz_ci_lower",
                "cohen_dz_ci_upper",
                "proportion_expected_direction",
                "wilcoxon_p_value",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
