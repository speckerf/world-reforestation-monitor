"""
Phase 1: Data Collection & Preprocessing (Orchestrator)

Run all three Phase 1 subscripts in sequence:
1. prepare_lut_domain.py - Assemble PROSAIL LUT Domain
2. prepare_rms_domain.py - Extract Grounded-EO Reference Measurements (RMs) Spectra
3. fetch_global_s2_domain.py - Sample Global Sentinel-2 Domain
"""

from __future__ import annotations

from pathlib import Path

from analysis.domain_representativeness_experiment.fetch_global_s2_domain import (
    fetch_global_s2_domain_simple,
)
from analysis.domain_representativeness_experiment.prepare_lut_domain import (
    prepare_lut_domain,
)
from analysis.domain_representativeness_experiment.prepare_rms_domain import (
    prepare_gbov_domain,
)


def run_phase1(base_dir: Path | None = None):
    """
    Execute Phase 1: Data Collection & Preprocessing.

    Parameters
    ----------
    base_dir : Path or None
        Output base directory (default: script location)
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("PHASE 1: DATA COLLECTION & PREPROCESSING")
    print("=" * 70)

    # Phase 1.1: Assemble PROSAIL LUT Domain
    print("\n" + "-" * 70)
    print("Phase 1.1: Assemble PROSAIL LUT Domain")
    print("-" * 70)
    lut_domains = prepare_lut_domain(base_dir=base_dir)

    # Phase 1.2: Extract Grounded-EO Reference Measurements (RMs) Spectra
    print("\n" + "-" * 70)
    print("Phase 1.2: Extract Grounded-EO Reference Measurements (RMs) Spectra")
    print("-" * 70)
    rms_domains = prepare_gbov_domain(base_dir=base_dir)

    # Phase 1.3: Sample Global Sentinel-2 Domain
    print("\n" + "-" * 70)
    print("Phase 1.3: Sample Global Sentinel-2 Domain")
    print("-" * 70)
    global_s2 = fetch_global_s2_domain_simple(
        year=2023, n_samples=5000, base_dir=base_dir
    )

    # Summary Report
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    data_dir = base_dir / "data"

    for trait in ["laie", "fapar", "fcover"]:
        print(f"\n{trait.upper()}:")
        print(f"  LUT domain:              {lut_domains[trait].shape}")
        print(f"  Grounded-EO RMs domain: {rms_domains[trait].shape}")

    print(f"\nGlobal S2 domain:       {global_s2.shape}")

    print(f"\nOutput files saved to: {data_dir}/")
    print("  - lut_combined_*.csv")
    print("  - grounded_eo_rms_spectra_*.csv")
    print("  - global_s2_spectra_synthetic_*.csv")

    # Display band information
    print("\nSpectral Bands (10 Sentinel-2 bands):")
    print("  B2 (Blue), B3 (Green), B4 (Red), B5-B7 (Red Edge)")
    print("  B8 (NIR), B8A (Red Edge), B11-B12 (SWIR)")

    print("\n" + "=" * 70)
    print("Phase 1 Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review output files in data/ folder")
    print("  2. Check reflectance scales (consistency between LUT and Grounded-EO RMs)")
    print("  3. Proceed to Phase 2: Simple 2D PCA Analysis")
    print("     Run: python run_phase2_pca.py")


if __name__ == "__main__":
    run_phase1()
