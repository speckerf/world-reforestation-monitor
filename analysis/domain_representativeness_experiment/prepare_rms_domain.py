"""Phase 1.2: Extract Grounded-EO Reference Measurements (RMs) Spectra

Extract 10 Sentinel-2 bands (B2-B8A, B11-B12) from GROUNDED-EO validation dataset.

Output: RMs spectral dataset (N_RMs × 10 bands) for each trait
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from train_pipeline.utils_loading import load_grounded_eo_validation_data

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


def prepare_gbov_domain(base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Extract Grounded-EO reference measurement (RM) spectra and organize by trait.

    Parameters
    ----------
    base_dir : Path or None
        Output base directory

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping trait -> Grounded-EO RMs spectra (N_rms × 10 bands)
    """
    data_dir, figures_dir = get_output_dirs(base_dir)

    print("Loading GROUNDED-EO validation data...")
    val_data = load_grounded_eo_validation_data()

    print(f"Loaded validation data with shape: {val_data.shape}")
    print(f"Columns: {list(val_data.columns[:20])}")  # Print first 20 columns

    # subset bandnames only / divide by 10000 to get reflectance values in [0,1]
    val_data = val_data[BAND_NAMES]
    val_data = val_data / 10000.0

    # save for later use
    rm_spectra_path = data_dir / "grounded_eo_rms_spectra.csv"
    val_data.to_csv(rm_spectra_path, index=False)
    print(f"Saved Grounded-EO RMs spectra to: {rm_spectra_path}")


if __name__ == "__main__":
    rms_domains = prepare_gbov_domain()
    print("\nPhase 1.2 complete: Grounded-EO RMs reference spectra extracted")
