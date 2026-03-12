"""
Phase 1.1: Assemble PROSAIL LUT Domain

Load trained LUT spectra from S2BIOPHYS models (5 ensemble members) and combine.
Extract 10 Sentinel-2 bands (B2-B8A, B11-B12).

Output: Combined LUT spectral dataset (N_luts × 10 bands)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

# Band mapping for Sentinel-2
BAND_NAMES = [
    "B2",  # 10m - Blue (490 nm)
    "B3",  # 10m - Green (560 nm)
    "B4",  # 10m - Red (665 nm)
    "B5",  # 20m - Vegetation Red Edge (705 nm)
    "B6",  # 20m - Vegetation Red Edge (740 nm)
    "B7",  # 20m - Vegetation Red Edge (783 nm)
    "B8",  # 10m - NIR (842 nm)
    "B8A",  # 20m - Vegetation Red Edge (865 nm)
    "B11",  # 20m - SWIR (1610 nm)
    "B12",  # 20m - SWIR (2190 nm)
]

TRAITS = ["laie", "fapar", "fcover"]


def get_output_dirs(base_dir: Path | None = None) -> tuple[Path, Path]:
    """Returns (data_dir, figures_dir). If base_dir is None, uses script directory."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figures_dir = base_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figures_dir


def load_lut_for_trait(
    trait: str | List[str],
    ensemble_size: int = 5,
    sample_lut_prop: float = 1.0,
    sample_lut_seed: int = 42,
) -> pd.DataFrame:
    """
    Load LUT files for a given trait across all ensemble members.

    Parameters
    ----------
    trait : str
        Trait name (laie, fapar, fcover)
    ensemble_size : int
        Number of ensemble members (typically 5)
    sample_lut_prop : float
        Proportion of LUT samples to use (default is 1.0, i.e., use all samples)

    Returns
    -------
    pd.DataFrame
        Combined LUT data with columns: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    """
    # Construct path to LUT files
    # data/train_pipeline/output/models/laie/lut_optuna-v2-laie-mlp-split-{i}.csv
    project_root = Path(__file__).resolve().parents[2]

    lut_dfs = []

    if isinstance(trait, str):
        trait = [trait]

    for t in trait:
        lut_base_dir = (
            project_root / "data" / "train_pipeline" / "output" / "models" / t
        )

        for i in range(ensemble_size):
            lut_file = lut_base_dir / f"lut_optuna-v2-{t}-mlp-split-{i}.csv"

            if not lut_file.exists():
                # print(f"  Warning: LUT file not found: {lut_file}, skipping")
                logger.warning(f"LUT file not found: {lut_file}, skipping")
                continue

            # print(f"  Loading {lut_file.name}")
            df = pd.read_csv(lut_file)

            # Extract 10 Sentinel-2 bands
            spectra_cols = [col for col in df.columns if col in BAND_NAMES]
            if len(spectra_cols) != 10:
                logger.warning(
                    f"    Warning: Expected 10 bands in LUT, found {len(spectra_cols)}: {spectra_cols}"
                )

            # Select only the bands we need, in correct order
            spectra_cols_ordered = [col for col in BAND_NAMES if col in df.columns]
            spectra = df[spectra_cols_ordered].copy()

            # Sample LUT spectra if sample_lut_prop < 1.0
            if sample_lut_prop < 1.0:
                n_samples = int(len(spectra) * sample_lut_prop)
                spectra = spectra.sample(n=n_samples, random_state=sample_lut_seed)

            lut_dfs.append(spectra)

        if not lut_dfs:
            raise ValueError(f"No valid LUT files found for trait: {t}")

    # Combine all ensemble members
    combined_lut = pd.concat(lut_dfs, ignore_index=True)

    return combined_lut


def prepare_lut_domain(base_dir: Path | None = None) -> dict:
    """
    Assemble PROSAIL LUT domain for all traits.

    Parameters
    ----------
    base_dir : Path or None
        Output base directory

    Returns
    -------
    dict
        Dictionary mapping trait -> LUT spectra (N_luts × 10 bands)
    """
    data_dir, figures_dir = get_output_dirs(base_dir)

    lut_domains = {}

    for trait in TRAITS:
        logger.trace(f"\nLoading LUT domain for {trait.upper()}:")

        lut_spectra = load_lut_for_trait(trait, ensemble_size=5)

        # exclude all with reflectance values outside [0,1] (if any)
        lut_spectra = lut_spectra[
            (lut_spectra >= 0).all(axis=1) & (lut_spectra <= 1).all(axis=1)
        ]

        logger.trace(f"  Combined LUT shape: {lut_spectra.shape}")
        logger.trace(f"  Min reflectance: {lut_spectra.min().min():.4f}")
        logger.trace(f"  Max reflectance: {lut_spectra.max().max():.4f}")
        logger.trace(f"  Mean reflectance: {lut_spectra.mean().mean():.4f}")

        lut_domains[trait] = lut_spectra

        # Save combined LUT
        lut_csv = data_dir / f"lut_combined_{trait}.csv"
        lut_spectra.to_csv(lut_csv, index=False)
        logger.info(f"  Saved to: {lut_csv}")

    return lut_domains


if __name__ == "__main__":
    lut_domains = prepare_lut_domain()

    logger.info("\nPhase 1.1 complete: PROSAIL LUT domain assembled")
