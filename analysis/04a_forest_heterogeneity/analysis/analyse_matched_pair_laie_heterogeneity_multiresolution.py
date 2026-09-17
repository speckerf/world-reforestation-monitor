#!/usr/bin/env python3
"""Compare LAIe heterogeneity in matched 500 m windows at 20 and 100 m."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds


METRICS = ("mean", "sd", "adjacent_difference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--laie-20m", type=Path, required=True)
    parser.add_argument("--laie-100m", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Long-format CSV: one row per pair and resolution")
    parser.add_argument("--summary-output", type=Path, required=True,
                        help="Resolution-effect summary with bootstrap intervals")
    parser.add_argument("--layer", default="matched_pairs")
    parser.add_argument("--window-size-m", type=float, default=500)
    parser.add_argument("--scale-factor-20m", type=float, default=1.0)
    parser.add_argument("--scale-factor-100m", type=float, default=1.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def window_metrics(array: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(array)
    x = array[valid]
    if x.size < 2:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "adjacent_difference": np.nan,
            "n_valid": int(x.size),
            "n_total": int(array.size),
            "valid_fraction": float(x.size / array.size),
        }

    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    differences = np.concatenate((
        np.abs(array[:, 1:] - array[:, :-1])[horizontal],
        np.abs(array[1:, :] - array[:-1, :])[vertical],
    ))
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "adjacent_difference": (
            float(np.mean(differences)) if differences.size else np.nan
        ),
        "n_valid": int(x.size),
        "n_total": int(array.size),
        "valid_fraction": float(x.size / array.size),
    }


def read_geographic_window(
    src: rasterio.io.DatasetReader,
    *,
    center_x: float,
    center_y: float,
    window_size_m: float,
    scale_factor: float,
) -> np.ndarray:
    """Read pixels whose centres fall inside an exact map-coordinate window."""
    if not src.crs or not src.crs.is_projected:
        raise ValueError("LAIe rasters must use a projected CRS.")
    if not np.isclose(src.transform.b, 0) or not np.isclose(src.transform.d, 0):
        raise ValueError("Rotated rasters are not supported.")
    if not np.isclose(abs(src.transform.a), abs(src.transform.e)):
        raise ValueError("LAIe rasters must have square pixels.")

    half = window_size_m / 2
    left, bottom = center_x - half, center_y - half
    right, top = center_x + half, center_y + half
    fractional = from_bounds(left, bottom, right, top, src.transform)
    col0 = max(0, int(np.floor(fractional.col_off)))
    row0 = max(0, int(np.floor(fractional.row_off)))
    col1 = min(src.width, int(np.ceil(fractional.col_off + fractional.width)))
    row1 = min(src.height, int(np.ceil(fractional.row_off + fractional.height)))
    candidate_window = Window(col0, row0, col1 - col0, row1 - row0)
    candidate = src.read(1, window=candidate_window, masked=True)

    cols = np.arange(col0, col1)
    rows = np.arange(row0, row1)
    xs = src.transform.c + (cols + 0.5) * src.transform.a
    ys = src.transform.f + (rows + 0.5) * src.transform.e
    tolerance = min(abs(src.transform.a), abs(src.transform.e)) * 1e-8
    keep_cols = (xs >= left - tolerance) & (xs < right - tolerance)
    keep_rows = (ys <= top + tolerance) & (ys > bottom + tolerance)
    selected = candidate[np.ix_(keep_rows, keep_cols)]

    expected = int(round(window_size_m / abs(src.transform.a)))
    if selected.shape != (expected, expected):
        raise ValueError(
            f"Expected {expected}x{expected} pixels for a {window_size_m:g} m "
            f"window at {abs(src.transform.a):g} m, obtained {selected.shape}. "
            "Check grid resolution, square pixels and raster coverage."
        )
    output = np.asarray(selected.filled(np.nan), dtype=np.float64)
    return output * scale_factor


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates = np.empty(samples)
    chunk_size = 2_000
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        boot = values[rng.integers(0, len(values), size=(stop - start, len(values)))]
        estimates[start:stop] = boot.mean(axis=1)
    return tuple(np.quantile(estimates, (0.025, 0.975)))


def create_summary(
    results: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for metric in METRICS:
        column = f"difference_{metric}"
        valid = results.loc[results["passes_valid_fraction"]]
        wide = valid.pivot(index="pair_id", columns="resolution_m", values=column)
        wide = wide.dropna(subset=[20, 100])
        if wide.empty:
            continue

        effects: dict[int, float] = {}
        for resolution in (20, 100):
            values = wide[resolution].to_numpy(float)
            lower, upper = bootstrap_mean_ci(
                values, samples=bootstrap_samples, rng=rng
            )
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            effects[resolution] = mean
            records.append({
                "metric": metric,
                "resolution_m": resolution,
                "n_pairs": len(values),
                "mean_difference": mean,
                "mean_ci_lower": lower,
                "mean_ci_upper": upper,
                "median_difference": float(np.median(values)),
                "cohen_dz": mean / sd if sd > 0 else np.nan,
                "proportion_plantation_lower": float(np.mean(values < 0)),
            })

        interaction = (wide[100] - wide[20]).to_numpy(float)
        interaction_lower, interaction_upper = bootstrap_mean_ci(
            interaction, samples=bootstrap_samples, rng=rng
        )
        attenuation = (
            1 - abs(effects[100]) / abs(effects[20])
            if effects[20] != 0 else np.nan
        )
        for record in records[-2:]:
            record.update({
                "attenuation_fraction": attenuation,
                "interaction_100m_minus_20m": float(interaction.mean()),
                "interaction_ci_lower": interaction_lower,
                "interaction_ci_upper": interaction_upper,
            })

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    if args.window_size_m <= 0:
        raise ValueError("window-size-m must be positive.")
    if not 0 <= args.min_valid_fraction <= 1:
        raise ValueError("min-valid-fraction must lie between 0 and 1.")
    if args.bootstrap_samples < 1_000:
        raise ValueError("Use at least 1,000 bootstrap samples.")

    pairs = gpd.read_file(args.pairs, layer=args.layer)
    required = {
        "pair_id", "control_class", "control_type", "distance_m",
        "plant_x", "plant_y", "control_x", "control_y",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairs file lacks columns: {sorted(missing)}")

    raster_specs = (
        (20, args.laie_20m, args.scale_factor_20m),
        (100, args.laie_100m, args.scale_factor_100m),
    )
    results: list[dict[str, object]] = []

    for resolution, raster_path, scale_factor in raster_specs:
        with rasterio.open(raster_path) as src:
            if pairs.crs is not None and src.crs != pairs.crs:
                raise ValueError(f"Pairs and {resolution} m LAIe use different CRSs.")
            actual_resolution = abs(src.transform.a)
            if not np.isclose(actual_resolution, resolution):
                raise ValueError(
                    f"Expected {resolution} m pixels in {raster_path}, "
                    f"found {actual_resolution:g} m."
                )

            for pair in pairs.itertuples(index=False):
                record: dict[str, object] = {
                    "pair_id": int(pair.pair_id),
                    "resolution_m": resolution,
                    "control_class": int(pair.control_class),
                    "control_type": pair.control_type,
                    "distance_m": float(pair.distance_m),
                }
                for prefix, x, y in (
                    ("plantation", pair.plant_x, pair.plant_y),
                    ("control", pair.control_x, pair.control_y),
                ):
                    array = read_geographic_window(
                        src,
                        center_x=float(x),
                        center_y=float(y),
                        window_size_m=args.window_size_m,
                        scale_factor=scale_factor,
                    )
                    for metric, value in window_metrics(array).items():
                        record[f"{prefix}_{metric}"] = value

                record["passes_valid_fraction"] = bool(
                    record["plantation_valid_fraction"] >= args.min_valid_fraction
                    and record["control_valid_fraction"] >= args.min_valid_fraction
                )
                for metric in METRICS:
                    record[f"difference_{metric}"] = (
                        record[f"plantation_{metric}"] - record[f"control_{metric}"]
                    )
                results.append(record)

    output = pd.DataFrame(results).sort_values(["pair_id", "resolution_m"])
    summary = create_summary(
        output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print(f"Saved {len(output):,} pair-resolution rows to {args.output}")
    print(f"Saved resolution summary to {args.summary_output}")
    print("\nPaired differences (plantation minus control):")
    print(summary.to_string(index=False))
    print(
        "\nNote: adjacent_difference represents a 20 m lag at 20 m resolution "
        "and a 100 m lag at 100 m resolution; do not interpret its attenuation "
        "as a pure resolution effect."
    )


if __name__ == "__main__":
    main()

