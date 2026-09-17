#!/usr/bin/env python3
"""Match plantation window centres to nearby non-plantation forest centres.

Input mask codes:
    1  = naturally regenerating forest
    10 = primary forest
    20 = plantation
    0  = invalid centre

The script runs randomized greedy searches and retains the solution with the
largest number of pairs (breaking ties by shorter total pair distance). All
centres belonging to different selected pairs are separated by at least the
requested exclusion distance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GeoPackage containing one line per pair.",
    )
    parser.add_argument("--max-distance-m", type=float, default=5_000)
    parser.add_argument("--exclusion-distance-m", type=float, default=1_000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--candidate-neighbours",
        type=int,
        default=64,
        help="Nearby controls considered per plantation centre.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def disk_structure(radius_pixels: int) -> np.ndarray:
    y, x = np.ogrid[
        -radius_pixels : radius_pixels + 1, -radius_pixels : radius_pixels + 1
    ]
    return (x * x + y * y) <= radius_pixels * radius_pixels


def mark_excluded(
    blocked: np.ndarray, row: int, col: int, structure: np.ndarray
) -> None:
    """Mark raster cells less than the exclusion distance from one centre."""
    radius = structure.shape[0] // 2
    r0, r1 = max(0, row - radius), min(blocked.shape[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(blocked.shape[1], col + radius + 1)
    sr0, sc0 = r0 - (row - radius), c0 - (col - radius)
    sr1, sc1 = sr0 + (r1 - r0), sc0 + (c1 - c0)
    blocked[r0:r1, c0:c1] |= structure[sr0:sr1, sc0:sc1]


def main() -> None:
    args = parse_args()
    if args.max_distance_m <= 0 or args.exclusion_distance_m <= 0:
        raise ValueError("Distances must be positive.")
    if args.iterations < 1 or args.candidate_neighbours < 1:
        raise ValueError("Iterations and candidate-neighbours must be >= 1.")

    with rasterio.open(args.mask) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        resolution = (abs(transform.a), abs(transform.e))

    if crs is None or not crs.is_projected:
        raise ValueError("Input mask must use a projected CRS with metre units.")
    if not np.isclose(resolution[0], resolution[1]):
        raise ValueError("Input mask must have square pixels.")

    plantation_rc = np.column_stack(np.where(mask == 20)).astype(np.int32)
    control_rc = np.column_stack(np.where(np.isin(mask, (1, 10)))).astype(np.int32)
    if plantation_rc.size == 0 or control_rc.size == 0:
        raise ValueError(
            "Mask must contain both plantation (20) and control (1/10) centres."
        )

    def coordinates(rc: np.ndarray) -> np.ndarray:
        xs, ys = xy(transform, rc[:, 0], rc[:, 1], offset="center")
        return np.column_stack((xs, ys))

    plantation_xy = coordinates(plantation_rc)
    control_xy = coordinates(control_rc)
    control_tree = cKDTree(control_xy)
    k = min(args.candidate_neighbours, len(control_rc))
    distances, neighbours = control_tree.query(
        plantation_xy, k=k, distance_upper_bound=args.max_distance_m
    )
    if k == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]

    has_candidate = np.isfinite(distances).any(axis=1)
    eligible_plantations = np.flatnonzero(has_candidate)
    if eligible_plantations.size == 0:
        raise ValueError("No plantation-control pairs occur within max-distance-m.")

    radius_pixels = int(np.ceil(args.exclusion_distance_m / resolution[0]))
    structure = disk_structure(radius_pixels)
    rng = np.random.default_rng(args.seed)
    best_pairs: list[tuple[int, int, float]] = []
    best_total_distance = np.inf

    for iteration in tqdm(range(args.iterations)):
        blocked = np.zeros(mask.shape, dtype=bool)
        pairs: list[tuple[int, int, float]] = []
        order = rng.permutation(eligible_plantations)

        for p_idx in order:
            pr, pc = plantation_rc[p_idx]
            if blocked[pr, pc]:
                continue

            valid = np.flatnonzero(np.isfinite(distances[p_idx]))
            # Randomize nearby alternatives so repeated runs explore different
            # greedy solutions, while retaining a mild preference for proximity.
            jitter = rng.random(valid.size) * args.max_distance_m * 0.10
            valid = valid[np.argsort(distances[p_idx, valid] + jitter)]

            chosen = None
            for rank in valid:
                c_idx = int(neighbours[p_idx, rank])
                cr, cc = control_rc[c_idx]
                if not blocked[cr, cc]:
                    chosen = c_idx
                    break
            if chosen is None:
                continue

            distance = float(np.linalg.norm(plantation_xy[p_idx] - control_xy[chosen]))
            pairs.append((int(p_idx), chosen, distance))
            mark_excluded(blocked, int(pr), int(pc), structure)
            cr, cc = control_rc[chosen]
            mark_excluded(blocked, int(cr), int(cc), structure)

        total_distance = sum(pair[2] for pair in pairs)
        if len(pairs) > len(best_pairs) or (
            len(pairs) == len(best_pairs) and total_distance < best_total_distance
        ):
            best_pairs = pairs
            best_total_distance = total_distance
        print(
            f"Iteration {iteration + 1:>3}/{args.iterations}: "
            f"{len(pairs)} pairs; best={len(best_pairs)}"
        )

    records = []
    for pair_id, (p_idx, c_idx, distance) in enumerate(best_pairs, start=1):
        pr, pc = plantation_rc[p_idx]
        cr, cc = control_rc[c_idx]
        pxy = plantation_xy[p_idx]
        cxy = control_xy[c_idx]
        control_class = int(mask[cr, cc])
        records.append(
            {
                "pair_id": pair_id,
                "control_class": control_class,
                "control_type": "primary"
                if control_class == 10
                else "naturally_regenerating",
                "distance_m": distance,
                "plant_row": int(pr),
                "plant_col": int(pc),
                "control_row": int(cr),
                "control_col": int(cc),
                "plant_x": float(pxy[0]),
                "plant_y": float(pxy[1]),
                "control_x": float(cxy[0]),
                "control_y": float(cxy[1]),
                "geometry": LineString([pxy, cxy]),
            }
        )

    output = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(args.output, layer="matched_pairs", driver="GPKG")
    print(f"Saved {len(output)} matched pairs to {args.output}")
    print(output["control_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
