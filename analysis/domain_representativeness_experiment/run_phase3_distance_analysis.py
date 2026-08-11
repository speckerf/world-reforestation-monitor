"""
Phase 3: Advanced Domain Analysis - Distance-to-Training-Domain
Analyze domain representativeness using distance metrics across all bands.

Key aspects:
- Compute KNN distances from RMs and Global S2 to LUT training domain
- Standardize in spectral domain (all 10 bands)
- Analyze distribution of distances
- Correlate performance with distance to training domain
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = "analysis/domain_representativeness_experiment"
DATA_DIR = f"{BASE_DIR}/data"
FIGURES_DIR = f"{BASE_DIR}/figures"
RESULTS_DIR = f"{BASE_DIR}/results"
N_NEIGHBORS = 5  # Use 5 nearest neighbors as mentioned in plan
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def load_domains():
    """Load all three spectral domains."""
    print("Loading spectral domains...")

    # Load LUT domain (training domain)
    lut_laie = pd.read_csv(f"{DATA_DIR}/lut_combined_laie.csv")
    # lut_fcover = pd.read_csv(f"{DATA_DIR}/lut_combined_fcover.csv")
    # lut_fapar = pd.read_csv(f"{DATA_DIR}/lut_combined_fapar.csv")

    # For main analysis, use LAIE (LAI-E) as primary model
    lut_domain = lut_laie[BAND_NAMES].values
    print(f"  LUT domain shape: {lut_domain.shape}")

    # Load Grounded-EO RMs domain (full data to preserve UUID)
    rms_full = pd.read_csv(f"{DATA_DIR}/grounded_eo_rms_spectra.csv")
    rms_uuid = rms_full.get("uuid", None)  # Try to get UUID if it exists

    # If UUID not in spectra file, try loading from original validation data
    if rms_uuid is None:
        print(
            "  UUID not found in spectra file, attempting to load from validation data..."
        )
        try:
            from train_pipeline.utils_loading import load_grounded_eo_validation_data

            full_val_data = load_grounded_eo_validation_data()
            rms_uuid = full_val_data["uuid"].values
            print(f"  Loaded UUID from validation data: {len(rms_uuid)} samples")
        except Exception as e:
            print(f"  Warning: Could not load UUID: {e}")
            rms_uuid = None
    else:
        rms_uuid = rms_uuid.values

    rms_domain = rms_full[BAND_NAMES].values
    print(f"  RMs domain shape: {rms_domain.shape}")

    # Load Global S2 domain
    global_s2 = pd.read_csv(f"{DATA_DIR}/global_s2_spectra_2020.csv")[BAND_NAMES].values
    global_s2 = global_s2 / 10000.0  # Scale to reflectance range

    print(f"  Global S2 domain shape: {global_s2.shape}")

    return lut_domain, rms_domain, global_s2, rms_uuid


def standardize_domains(lut_domain, rms_domain, global_s2):
    """Standardize all domains to zero mean, unit variance in spectral space."""
    print("\nStandardizing spectral domains...")

    # Combine all data for consistent standardization
    all_data = np.vstack([lut_domain, rms_domain, global_s2])

    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data)

    # Split back into domains
    n_lut = lut_domain.shape[0]
    n_rms = rms_domain.shape[0]

    lut_scaled = all_data_scaled[:n_lut]
    rms_scaled = all_data_scaled[n_lut : n_lut + n_rms]
    global_s2_scaled = all_data_scaled[n_lut + n_rms :]

    print(f"  Scaler mean: {scaler.mean_}")
    print(f"  Scaler std: {scaler.scale_}")

    return lut_scaled, rms_scaled, global_s2_scaled, scaler


def compute_knn_distances(reference_domain, query_domain, n_neighbors=N_NEIGHBORS):
    """
    Compute distances to nearest neighbors in reference domain.

    Parameters:
    -----------
    reference_domain : array (N_ref, n_bands)
        Reference training domain (LUT)
    query_domain : array (N_query, n_bands)
        Query domain (RMs or Global S2)
    n_neighbors : int
        Number of nearest neighbors to consider

    Returns:
    --------
    distances : array (N_query,)
        Average distance to k nearest neighbors
    distances_all : array (N_query, n_neighbors)
        All k distances for each query point
    indices : array (N_query, n_neighbors)
        Indices of k nearest neighbors
    """
    print(f"\n  Computing KNN distances (k={n_neighbors})...")

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="kd_tree", metric="manhattan"
    ).fit(reference_domain)
    distances_all, indices = nbrs.kneighbors(query_domain)

    # Average distance to k nearest neighbors
    distances = np.mean(distances_all, axis=1)

    print(f"  Mean distance: {np.mean(distances):.6f}")
    print(f"  Std distance: {np.std(distances):.6f}")
    print(f"  Min distance: {np.min(distances):.6f}")
    print(f"  Max distance: {np.max(distances):.6f}")
    print(
        f"  Percentiles: [10, 25, 50, 75, 90]: {np.percentile(distances, [10, 25, 50, 75, 90])}"
    )

    return distances, distances_all, indices


def analyze_domain_distances(lut_scaled, rms_scaled, global_s2_scaled):
    """Analyze distance distributions from training domain."""
    print("\n" + "=" * 70)
    print("DISTANCE ANALYSIS: Distance to LUT Training Domain")
    print("=" * 70)

    print("\n1. RMs Distance to LUT:")
    rms_distances, rms_distances_all, rms_indices = compute_knn_distances(
        lut_scaled, rms_scaled, N_NEIGHBORS
    )

    print("\n2. Global S2 Distance to LUT:")
    global_distances, global_distances_all, global_indices = compute_knn_distances(
        lut_scaled, global_s2_scaled, N_NEIGHBORS
    )

    return {
        "rms_distances": rms_distances,
        "rms_distances_all": rms_distances_all,
        "global_distances": global_distances,
        "global_distances_all": global_distances_all,
    }


def compute_cumulative_coverage(distances, distance_thresholds=None):
    """Compute cumulative distribution: % of points within distance D to LUT."""
    if distance_thresholds is None:
        distance_thresholds = np.arange(0, np.max(distances) + 0.01, 0.01)

    cumulative = [np.mean(distances <= d) * 100 for d in distance_thresholds]

    return distance_thresholds, cumulative


def plot_distance_distributions(analysis_results):
    """Create comprehensive distance distribution plots."""
    print("\nGenerating distance distribution plots...")

    rms_distances = analysis_results["rms_distances"]
    global_distances = analysis_results["global_distances"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Domain Representativeness: Distance-to-LUT Analysis\n(All 10 Sentinel-2 Bands)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Histogram comparison
    ax = axes[0, 0]
    bins = np.linspace(0, 10, 30)
    ax.hist(
        rms_distances,
        bins=bins,
        alpha=0.6,
        label=f"RMs (n={len(rms_distances)})",
        color="red",
        edgecolor="black",
    )
    ax.hist(
        global_distances,
        bins=bins,
        alpha=0.5,
        label=f"Global S2 (n={len(global_distances)})",
        color="green",
        edgecolor="black",
    )
    ax.set_xlabel("Distance to LUT (standardized Manhattan)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distance Distribution Histograms")
    ax.set_xlim(0, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: KDE density
    ax = axes[0, 1]
    x_range = np.linspace(0, max(np.max(rms_distances), np.max(global_distances)), 200)

    kde_rms = gaussian_kde(rms_distances)
    ax.plot(x_range, kde_rms(x_range), "r-", linewidth=2, label="RMs")
    ax.fill_between(x_range, kde_rms(x_range), alpha=0.3, color="red")

    kde_global = gaussian_kde(global_distances)
    ax.plot(x_range, kde_global(x_range), "g-", linewidth=2, label="Global S2")
    ax.fill_between(x_range, kde_global(x_range), alpha=0.3, color="green")

    ax.set_xlabel("Distance to LUT (standardized Manhattan)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Kernel Density Estimation")
    ax.set_xlim(0, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative distribution
    ax = axes[1, 0]
    rms_thresh, rms_cumul = compute_cumulative_coverage(rms_distances)
    global_thresh, global_cumul = compute_cumulative_coverage(global_distances)

    ax.plot(
        rms_thresh,
        rms_cumul,
        "r-",
        linewidth=2.5,
        label="RMs",
        marker="o",
        markersize=4,
        alpha=0.7,
    )
    ax.plot(
        global_thresh,
        global_cumul,
        "g-",
        linewidth=2.5,
        label="Global S2",
        marker="s",
        markersize=4,
        alpha=0.7,
    )
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% Coverage")
    ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5, label="90% Coverage")
    ax.set_xlabel("Distance Threshold (standardized Manhattan)", fontsize=11)
    ax.set_ylabel("Cumulative % Within Distance", fontsize=11)
    ax.set_title("Cumulative Coverage Analysis")
    ax.set_xlim(0, 10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Box plot comparison
    ax = axes[1, 1]
    data_to_plot = [rms_distances, global_distances]
    bp = ax.boxplot(data_to_plot, labels=["RMs", "Global S2"], patch_artist=True)
    bp["boxes"][0].set_facecolor("red")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("green")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("Distance to LUT (standardized Manhattan)", fontsize=11)
    ax.set_title("Distance Distribution Summary")
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        f"{FIGURES_DIR}/phase3_distance_distributions.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {FIGURES_DIR}/phase3_distance_distributions.png")
    plt.close()


def plot_distance_by_neighbor(analysis_results):
    """Plot distance to 1st, 2nd, 3rd, 4th, 5th nearest neighbors."""
    print("Generating per-neighbor distance plots...")

    rms_distances_all = analysis_results["rms_distances_all"]
    global_distances_all = analysis_results["global_distances_all"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Distance to K-th Nearest Neighbor in LUT Domain",
        fontsize=13,
        fontweight="bold",
    )

    # RMs distances by neighbor rank
    ax = axes[0]
    for k in range(N_NEIGHBORS):
        ax.plot(
            np.sort(rms_distances_all[:, k]),
            label=f"{k + 1}NN",
            alpha=0.7,
            linewidth=1.5,
        )
    ax.set_xlabel("Sample Index (sorted)", fontsize=11)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title(f"RMs (n={len(rms_distances_all)})")
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Global S2 distances by neighbor rank
    ax = axes[1]
    for k in range(N_NEIGHBORS):
        ax.plot(
            np.sort(global_distances_all[:, k]),
            label=f"{k + 1}NN",
            alpha=0.7,
            linewidth=1.5,
        )
    ax.set_xlabel("Sample Index (sorted)", fontsize=11)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title(f"Global S2 (n={len(global_distances_all)})")
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{FIGURES_DIR}/phase3_neighbor_distances.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {FIGURES_DIR}/phase3_neighbor_distances.png")
    plt.close()


def save_distance_statistics(analysis_results):
    """Save detailed distance statistics to CSV."""
    print("\nSaving distance statistics...")

    rms_distances = analysis_results["rms_distances"]
    global_distances = analysis_results["global_distances"]

    stats = {
        "Domain": ["RMs", "Global S2"],
        "N_samples": [len(rms_distances), len(global_distances)],
        "Mean_distance": [np.mean(rms_distances), np.mean(global_distances)],
        "Std_distance": [np.std(rms_distances), np.std(global_distances)],
        "Min_distance": [np.min(rms_distances), np.min(global_distances)],
        "Max_distance": [np.max(rms_distances), np.max(global_distances)],
        "P10_distance": [
            np.percentile(rms_distances, 10),
            np.percentile(global_distances, 10),
        ],
        "P25_distance": [
            np.percentile(rms_distances, 25),
            np.percentile(global_distances, 25),
        ],
        "P50_distance": [
            np.percentile(rms_distances, 50),
            np.percentile(global_distances, 50),
        ],
        "P75_distance": [
            np.percentile(rms_distances, 75),
            np.percentile(global_distances, 75),
        ],
        "P90_distance": [
            np.percentile(rms_distances, 90),
            np.percentile(global_distances, 90),
        ],
    }

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f"{RESULTS_DIR}/phase3_distance_statistics.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/phase3_distance_statistics.csv")
    print("\nDistance Statistics Summary:")
    print(stats_df.to_string(index=False))

    return stats_df


def save_distance_data(
    analysis_results, lut_scaled, rms_scaled, global_s2_scaled, rms_uuid=None
):
    """Save detailed distance data with coordinates for further analysis."""
    print("\nSaving detailed distance data...")

    rms_distances = analysis_results["rms_distances"]
    global_distances = analysis_results["global_distances"]

    # RMs with distances
    rms_output = pd.DataFrame(
        rms_scaled,
        columns=[f"B{i + 2}" if i < 8 else f"B{[11, 12][i - 8]}" for i in range(10)],
    )
    rms_output["distance_to_lut"] = rms_distances

    # Add UUID if available
    if rms_uuid is not None:
        if len(rms_uuid) == len(rms_output):
            rms_output["uuid"] = rms_uuid
            print(f"  UUID column added ({len(rms_uuid)} entries)")
        else:
            print(
                f"  Warning: UUID length ({len(rms_uuid)}) doesn't match RMs ({len(rms_output)})"
            )

    rms_output.to_csv(f"{RESULTS_DIR}/phase3_rms_with_distances.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/phase3_rms_with_distances.csv")

    # Global S2 with distances
    global_output = pd.DataFrame(
        global_s2_scaled,
        columns=[f"B{i + 2}" if i < 8 else f"B{[11, 12][i - 8]}" for i in range(10)],
    )
    global_output["distance_to_lut"] = global_distances
    global_output.to_csv(
        f"{RESULTS_DIR}/phase3_globalS2_with_distances.csv", index=False
    )
    print(f"  Saved: {RESULTS_DIR}/phase3_globalS2_with_distances.csv")


def main():
    """Run Phase 3 distance-to-training-domain analysis."""
    print("\n" + "=" * 70)
    print("PHASE 3: ADVANCED DOMAIN ANALYSIS - DISTANCE-TO-TRAINING-DOMAIN")
    print("=" * 70)

    # Load domains
    lut_domain, rms_domain, global_s2, rms_uuid = load_domains()

    # Standardize all domains
    lut_scaled, rms_scaled, global_s2_scaled, scaler = standardize_domains(
        lut_domain, rms_domain, global_s2
    )

    # Analyze distances
    analysis_results = analyze_domain_distances(
        lut_scaled, rms_scaled, global_s2_scaled
    )

    # Visualizations
    plot_distance_distributions(analysis_results)
    plot_distance_by_neighbor(analysis_results)

    # Save results
    save_distance_statistics(analysis_results)
    save_distance_data(
        analysis_results, lut_scaled, rms_scaled, global_s2_scaled, rms_uuid
    )

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review distance distribution plots")
    print("  2. Compare RMs vs Global S2 domain coverage")
    print("  3. Identify regions/points far from training domain")
    print("  4. Correlate with model performance (if available)")
    print("  5. Generate phase 3 report with findings")


if __name__ == "__main__":
    main()
    # main()
    # main()
    # main()
    # main()
    # main()
    # main()
    # main()
    # main()
