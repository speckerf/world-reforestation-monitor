"""
Phase 2: Simple 2D PCA Analysis

Compare domain representativeness across three spectral domains:
1. PROSAIL LUT (training domain)
2. Grounded-EO RMs (ground truth validation)
3. Global Sentinel-2 (operational domain)

Creates scatter plots with 2D PCA projections.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TRAITS = ["laie", "fapar", "fcover"]

# Sentinel-2 band names
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def load_phase1_data(data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """
    Load all Phase 1 outputs for analysis.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Structure: {trait: {"lut": array, "rms": array, "global_s2": array}}
    """
    data = {}

    for trait in TRAITS:
        lut_file = data_dir / f"lut_combined_{trait}.csv"
        rms_file = data_dir / "grounded_eo_rms_spectra.csv"
        s2_file = data_dir / "global_s2_spectra_2020.csv"

        # Load LUT
        lut_df = pd.read_csv(lut_file)

        # Load RMs
        rms_df = pd.read_csv(rms_file)

        # load global S2 reflectances:
        s2_df = pd.read_csv(s2_file)[BAND_NAMES]

        # assert column order matches
        assert list(lut_df.columns) == BAND_NAMES, (
            f"LUT columns mismatch: {lut_df.columns}"
        )
        assert list(rms_df.columns) == BAND_NAMES, (
            f"RMs columns mismatch: {rms_df.columns}"
        )

        assert list(s2_df.columns) == BAND_NAMES, (
            f"Global S2 columns mismatch: {s2_df.columns}"
        )

        # divide s2_df by 10000 to get reflectance values in [0,1]
        s2_df = s2_df / 10000.0

        data[trait] = {
            "lut_df": lut_df,
            "rms_df": rms_df,
            "s2_df": s2_df,
        }

    return data


def perform_2d_pca(
    data: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Perform 2D PCA for each trait combining LUT + RMs domains.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Structure: {trait: {"lut_pca": array (N,2), "rms_pca": array (M,2),
                            "pca": PCA, "scaler": StandardScaler}}
    """
    pca_results = {}

    for trait in TRAITS:
        # Combine LUT + RMs for fitting PCA
        lut = data[trait]["lut_df"].values
        rms = data[trait]["rms_df"].values
        s2 = data[trait]["s2_df"].values
        combined = np.vstack([lut, rms, s2])

        # Standardize
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)

        # Fit PCA (10 components)
        pca = PCA(n_components=10)
        pca.fit(combined_scaled)

        # Transform both domains
        lut_scaled = scaler.transform(lut)
        rms_scaled = scaler.transform(rms)
        s2_scaled = scaler.transform(s2)

        lut_pca = pca.transform(lut_scaled)
        rms_pca = pca.transform(rms_scaled)
        s2_pca = pca.transform(s2_scaled)

        pca_results[trait] = {
            "lut_pca": lut_pca,
            "rms_pca": rms_pca,
            "s2_pca": s2_pca,
            "pca": pca,
            "scaler": scaler,
            "combined": combined_scaled,
        }

        # Print explained variance
        print(f"\n{trait.upper()} - Explained Variance:")
        print(
            f"  PC1: {pca.explained_variance_ratio_[0]:.2%}, "
            f"PC2: {pca.explained_variance_ratio_[1]:.2%}"
        )
        print(f"  Total: {sum(pca.explained_variance_ratio_):.2%}sta")

    return pca_results


def subsample_points(
    pca_results: dict[str, dict[str, np.ndarray]],
    n_samples: int = 3000,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Subsample LUT points to reduce visual clutter in plots.

    Parameters
    ----------
    pca_results : dict
        PCA results from perform_2d_pca()
    n_samples : int
        Maximum number of LUT samples to keep (default: 3000)

    Returns
    -------
    dict
        PCA results with subsampled LUT points
    """
    subsampled = {}

    for trait in TRAITS:
        lut_pca = pca_results[trait]["lut_pca"]
        rms_pca = pca_results[trait]["rms_pca"]
        s2_pca = pca_results[trait]["s2_pca"]

        # Subsample LUT if it exceeds n_samples
        if len(lut_pca) > n_samples:
            indices = np.random.choice(len(lut_pca), n_samples, replace=False)
            lut_pca_sub = lut_pca[indices]
            print(
                f"  {trait.upper()}: Subsampled LUT from {len(lut_pca)} to {n_samples} points"
            )
        else:
            lut_pca_sub = lut_pca
            print(
                f"  {trait.upper()}: LUT has {len(lut_pca)} points (below {n_samples} threshold)"
            )

        if len(rms_pca) > n_samples:
            indices = np.random.choice(len(rms_pca), n_samples, replace=False)
            rms_pca_sub = rms_pca[indices]
            print(
                f"  {trait.upper()}: RMs has {len(rms_pca)} points (subsampled to {n_samples} points)"
            )
        else:
            rms_pca_sub = rms_pca
            print(
                f"  {trait.upper()}: RMs has {len(rms_pca)} points (below {n_samples} threshold)"
            )

        if len(s2_pca) > n_samples:
            indices = np.random.choice(len(s2_pca), n_samples, replace=False)
            s2_pca_sub = s2_pca[indices]
            print(
                f"  {trait.upper()}: Global S2 has {len(s2_pca)} points (subsampled to {n_samples} points)"
            )
        else:
            s2_pca_sub = s2_pca
            print(
                f"  {trait.upper()}: Global S2 has {len(s2_pca)} points (below {n_samples} threshold)"
            )

        subsampled[trait] = {
            **pca_results[trait],
            "lut_pca": lut_pca_sub,
            "rms_pca": rms_pca_sub,
            "s2_pca": s2_pca_sub,
        }

    return subsampled


def create_pca_plots(
    pca_results: dict[str, dict[str, np.ndarray]],
    figures_dir: Path,
    n_samples: int = 3000,
    pc1_idx: int = 0,
    pc2_idx: int = 1,
) -> None:
    """Create visualization of 2D PCA projections with subsampling.

    Parameters
    ----------
    pca_results : dict
        PCA results from perform_2d_pca()
    figures_dir : Path
        Directory to save figures
    n_samples : int
        Maximum number of LUT samples to display (default: 3000)
    """
    # Subsample for cleaner visualization
    pca_results_sub = subsample_points(pca_results, n_samples=n_samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Phase 2: 2D PCA Domain Representativeness Analysis",
        fontsize=14,
        fontweight="bold",
    )

    for idx, trait in enumerate(TRAITS):
        ax = axes[idx]

        lut_pca = pca_results_sub[trait]["lut_pca"]
        rms_pca = pca_results_sub[trait]["rms_pca"]
        s2_pca = pca_results_sub[trait]["s2_pca"]
        pca = pca_results_sub[trait]["pca"]

        # Plot LUT points (smaller, more transparent)
        ax.scatter(
            lut_pca[:, pc1_idx],
            lut_pca[:, pc2_idx],
            alpha=0.2,
            s=3,
            c="blue",
            label="PROSAIL LUT",
        )

        # Plot RMs points (small, distinctive)
        ax.scatter(
            rms_pca[:, pc1_idx],
            rms_pca[:, pc2_idx],
            alpha=0.2,
            s=3,
            c="red",
            label="Grounded-EO RMs",
            edgecolors="darkred",
            linewidth=0.3,
        )

        # Plot S2 points (small, distinctive)
        ax.scatter(
            s2_pca[:, pc1_idx],
            s2_pca[:, pc2_idx],
            alpha=0.2,
            s=3,
            c="green",
            label="Global S2",
            edgecolors="darkgreen",
            linewidth=0.3,
        )

        ax.set_xlabel(
            f"PC{pc1_idx + 1} ({pca.explained_variance_ratio_[pc1_idx]:.1%})",
            fontsize=10,
        )
        ax.set_ylabel(
            f"PC{pc2_idx + 1} ({pca.explained_variance_ratio_[pc2_idx]:.1%})",
            fontsize=10,
        )
        ax.set_title(f"{trait.upper()}", fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = figures_dir / f"phase2_pca_2d_pc{pc1_idx + 1}_pc{pc2_idx + 1}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_file}")
    plt.close()


def run_phase2(base_dir: Path | None = None) -> None:
    """Execute Phase 2: 2D PCA Analysis."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    data_dir = base_dir / "data"
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: SIMPLE 2D PCA ANALYSIS")
    print("=" * 70)

    # Load Phase 1 outputs
    print("\nLoading Phase 1 data...")
    data = load_phase1_data(data_dir)

    # Perform PCA
    print("\nPerforming 2D PCA for each trait...")
    pca_results = perform_2d_pca(data)

    # Create visualizations
    print("\nCreating visualizations...")
    print("Subsampling LUT for cleaner visualization:")
    create_pca_plots(pca_results, figures_dir, n_samples=3000, pc1_idx=0, pc2_idx=1)
    create_pca_plots(pca_results, figures_dir, n_samples=3000, pc1_idx=1, pc2_idx=2)
    create_pca_plots(pca_results, figures_dir, n_samples=3000, pc1_idx=0, pc2_idx=2)

    # Summary report
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)

    print("\n2D PCA plots created for all traits (LAIe, FAPAR, FCOVER)")
    print("  - Blue points: PROSAIL LUT samples (training domain)")
    print("  - Red points: Grounded-EO RMs samples (validation domain)")
    print("  - Green points: Global S2 samples (additional domain)")
    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {figures_dir}/phase2_pca_2d_pc1_pc2.png")
    print(f"  - {figures_dir}/phase2_pca_2d_pc1_pc3.png")

    print(f"  - {figures_dir}/phase2_explained_variance.png")
    print("\nNext steps:")
    print("  - Review scatter plots to identify domain gaps")
    print("  - Proceed to Phase 3: Advanced Domain Analysis (if needed)")


if __name__ == "__main__":
    run_phase2()
