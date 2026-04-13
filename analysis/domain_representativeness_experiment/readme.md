Reviewer comment: 

Lines 533-534: Although the strategy is valuable, it does not adequately address the "bubble gaps" that occur within the convex hull of the training data. Please analyze how the GBOV reference set overlaps with your simulated parameter space and identify areas where it falls outside of it. For instance, you can represent your S2 information in a 2D space using the first two PCA components. This is a common practice for demonstrating the representativeness of ground truth observations. This analysis will help clarify the limitations of your model-parameter optimization and highlight any regions where performance may deteriorate.


## Quick Start: Phase 3 Distance Analysis

Phase 3 implements distance-to-training-domain analysis using all 10 Sentinel-2 bands (not PCA-reduced).

### Run Sequential Steps

```bash
cd analysis/domain_representativeness_experiment

# Phase 3.1: Compute distances from RMs and Global S2 to LUT training domain
python run_phase3_distance_analysis.py

# Phase 3.2: Get S2BIOPHYS predictions and analyze performance vs distance
python run_phase3_performance_vs_distance.py
```

### Expected Outputs

From Phase 3.1:
- `results/phase3_distance_statistics.csv` - Summary statistics
- `results/phase3_rms_with_distances.csv` - RMs coordinates + distances (used by Phase 3.2)
- `figures/phase3_distance_distributions.png` - 4-panel distribution analysis
- `figures/phase3_neighbor_distances.png` - Per-neighbor rank analysis

From Phase 3.2:
- `results/predictions_rms_phase3.csv` - RMs with predictions, errors, uncertainty
- `results/performance_by_distance_phase3.csv` - Binned performance by distance
- `figures/phase3_performance_vs_distance.png` - Performance degradation with distance
- `figures/phase3_uncertainty_vs_distance.png` - Uncertainty vs distance

### Key Insights to Extract

- **Distance coverage**: % of RMs and Global S2 within each distance threshold to LUT
- **Performance degradation**: How MAE/RMSE increases with distance from training domain
- **Uncertainty calibration**: Does ensemble uncertainty grow with distance?
- **OOD samples**: Identify RMs far from LUT space where model extrapolates

---

## Implementation

Overview: Create a high-level comparison of the expected domain representativness between simulated training data from PROSAIL (from S2BIOPHYS model), with added reflectance spectra from OSSL, and EMIT (according to the  model specification; e.g. the number of added emit spectra etc.). Use qualitative assessment based on 2D PCA of the 10 bands. 

For the training LUT, it is important that we reproduce the LUT by combining all simualted with non-vegetated spectra like (depending on the model specification (watch out each model ensemble member is different in S2bIOPHYS)). 

Specifically i want to analyze 3-fold interactions: overlap of calibration points (reference measurements) with PROSAIL LUT used for training and with global sample of random Sentinel-2 reflectances. 

To obtain global sample of random Sentinel-2 reflectances: create a separaet script that uses python api to sample s2 reflectances. Specifically, fetch all images from a given year (using s2 system indices in `data/gee_pipeline/outputs/s2_indices_per_mgrs_tile`) using filter. First filter year; then filter system index; then pait with cloud score polus to mask clouds; then mosaic; then mask inland waters (check gee_pipeline.srcGlobal) and ecoregion not equal to 0 (check gee_pipeline.srcGlobal) and the extract points. Points shall be taken from the following feature collection: `ee.FeatureCollection('projects/ee-speckerfelix/assets/open-earth/fibonacci-samples')`. 

First strategy is a easy 2D PCA between all thre way combinations. Then in a next step think about the  more complex approach: Compare this domain (either using a reflectnace hypercube grid; so set min max for each band; then divide into 10 bins; then assess how many of these spaces are accopied, or whats the average distance to the training domain)


Extrapolation Map: 100m Global map of average distance to LUT; 
- Create random selection of all LUT values (laie, fapar, and fcover?)
- Repeat 5 times:
    - For each MGRS tile; create random S2 mosaic from selected scenes (peak growing season)
        - sort using randomColumn before mosaicking
        - mask clouds / mask waters / mask ecoregion 0
    - Stream 100m S2 mosaic using Xee to local; 
    - Build nearest neighbour classifier using XDtree / use Manhattan distance
    - Calculate avg. distance to LUT
- Average from 5 seeds; Save to local file again; tif file / save as uint8 (nodata value 255)
- Upload local tif file vis GCS to Google Assets


