#!/usr/bin/env python3
"""Calculate LAIe heterogeneity within matched 500 m forest windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# METRICS = ("mean", "sd", "rao_q", "adjacent_difference", "n_valid", "valid_fraction")
# METRICS = ("mean", "sd", "adjacent_difference", "n_valid", "valid_fraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="GeoPackage produced by match_forest_window_centres.py",
    )
    parser.add_argument(
        "--laie",
        type=Path,
        required=True,
        help="20 m LAIe raster on the sampling-mask grid",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV with one row per matched pair",
    )
    parser.add_argument("--layer", default="matched_pairs")
    parser.add_argument("--window-pixels", type=int, default=25)
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Multiply raster values by this factor (e.g. 0.001)",
    )
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    return parser.parse_args()


# def rao_q_absolute(values: np.ndarray) -> float:
#     """Mean absolute difference across all ordered pairs, without an n² matrix."""
#     x = np.sort(values)
#     n = x.size
#     if n < 2:
#         return np.nan
#     weights = 2 * np.arange(1, n + 1) - n - 1
#     return float(2 * np.sum(weights * x) / n**2)


def window_metrics(array: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(array)
    x = array[valid]
    n_total = array.size
    if x.size < 2:
        return {
            "mean": np.nan,
            "sd": np.nan,
            # "rao_q": np.nan,
            "adjacent_difference": np.nan,
            "n_valid": int(x.size),
            "valid_fraction": float(x.size / n_total),
        }

    # Local spatial contrast from horizontal and vertical valid neighbours.
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    differences = np.concatenate(
        (
            np.abs(array[:, 1:] - array[:, :-1])[horizontal],
            np.abs(array[1:, :] - array[:-1, :])[vertical],
        )
    )

    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        # "rao_q": rao_q_absolute(x),
        "adjacent_difference": float(np.mean(differences))
        if differences.size
        else np.nan,
        "n_valid": int(x.size),
        "valid_fraction": float(x.size / n_total),
    }


def read_centered_window(
    src: rasterio.io.DatasetReader,
    row: int,
    col: int,
    size: int,
    scale_factor: float,
) -> np.ndarray:
    half = size // 2
    window = Window(col - half, row - half, size, size)
    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + size > src.width
        or window.row_off + size > src.height
    ):
        raise ValueError(f"Window centred at row={row}, col={col} exceeds LAIe bounds.")
    data = src.read(1, window=window, masked=True)
    output = np.asarray(data.filled(np.nan), dtype=np.float64)
    output *= scale_factor
    return output


def main() -> None:
    args = parse_args()
    if args.window_pixels < 3 or args.window_pixels % 2 == 0:
        raise ValueError("window-pixels must be an odd integer >= 3.")
    if not 0 <= args.min_valid_fraction <= 1:
        raise ValueError("min-valid-fraction must lie between 0 and 1.")

    pairs = gpd.read_file(args.pairs, layer=args.layer)
    required = {
        "pair_id",
        "control_class",
        "control_type",
        "plant_row",
        "plant_col",
        "control_row",
        "control_col",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairs file lacks columns: {sorted(missing)}")

    results: list[dict[str, object]] = []
    with rasterio.open(args.laie) as src:
        if pairs.crs is not None and src.crs != pairs.crs:
            raise ValueError("Pairs and LAIe raster use different coordinate systems.")

        for pair in pairs.itertuples(index=False):
            record: dict[str, object] = {
                "pair_id": pair.pair_id,
                "control_class": int(pair.control_class),
                "control_type": pair.control_type,
                "distance_m": float(pair.distance_m),
            }

            for prefix, row, col in (
                ("plantation", pair.plant_row, pair.plant_col),
                ("control", pair.control_row, pair.control_col),
            ):
                array = read_centered_window(
                    src,
                    int(row),
                    int(col),
                    args.window_pixels,
                    args.scale_factor,
                )
                metrics = window_metrics(array)
                for metric, value in metrics.items():
                    record[f"{prefix}_{metric}"] = value

            record["passes_valid_fraction"] = bool(
                record["plantation_valid_fraction"] >= args.min_valid_fraction
                and record["control_valid_fraction"] >= args.min_valid_fraction
            )
            for metric in ("mean", "sd", "adjacent_difference"):
                record[f"difference_{metric}"] = (
                    record[f"plantation_{metric}"] - record[f"control_{metric}"]
                )
            results.append(record)

    output = pd.DataFrame(results).sort_values("pair_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    valid = output.loc[output["passes_valid_fraction"]]
    print(f"Saved {len(output):,} pairs to {args.output}")
    print(f"Pairs passing valid-pixel threshold: {len(valid):,}")
    if len(valid):
        summary = valid.groupby("control_type")[
            [
                "difference_mean",
                "difference_sd",
                # "difference_rao_q",
                "difference_adjacent_difference",
            ]
        ].agg(["count", "mean", "median"])
        print("\nPaired differences (plantation minus control):")
        print(summary.to_string())


if __name__ == "__main__":
    main()
