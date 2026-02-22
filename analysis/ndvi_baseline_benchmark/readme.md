# NDVI Baseline Benchmark

## Reviewer Question

*Lines 493-496: Please include a benchmark in which LAIe, FAPAR, and FCOVER are retrieved with a straightforward non-linear NDVI-based model (e.g., an empirical non-linear NDVI-variable curve/model fitted to the GBOV data). This sanity check would clarify whether the added complexity of the hybrid PROSAIL-MLP pipeline provides a meaningful performance improvement over a much simpler approach.

## Our Response

We have added a benchmark against a simple empirical NDVI-based model, implemented as a third-degree polynomial NDVI–variable regression fitted to the GROUNDED-EO reference measurements. The NDVI model is evaluated using the same ecoregion-based spatially blocked cross-validation splits used in S2BIOPHYS training, ensuring a fair comparison under identical train/validation split structure. As shown in Table 1 (below), results show trait-dependent performance differences. S2BIOPHYS provides the largest improvement for LAIe (R² +0.052), modest improvement for FCOVER (R² +0.013), while NDVI actually achieves slightly better performance for FAPAR (R² +0.013). These results demonstrate that while S2BIOPHYS provides clear benefits for more complex variables like LAIe through its physical basis, empirical NDVI models can be equally competitive for simpler variables like FAPAR.

Beyond aggregate metrics, a key strength of S2BIOPHYS is its ability to produce physically consistent predictions across the full range of global surface conditions, including non-vegetated surfaces where LAIe, FAPAR, and FCOVER are expected to approach zero. In contrast, we don't expect an empirical NDVI-based polynomial model calibrated within the vegetation-dominated NDVI range of the reference data to extrapolate reliably to such conditions.

As this benchmark primarily serves as a sanity check, it is not included in the main manuscript to maintain focus on the proposed retrieval framework and comparison with existing algorithms. All code, results, and additional diagnostic figures of this sanity check are provided in the public code repository (see `analysis/ndvi_baseline_benchmark/`).

## Table 1: Test-set Performance Comparison

Comparison between a simple NDVI-based polynomial baseline and the hybrid PROSAIL–MLP (S2BIOPHYS) model.

| Variable | Model | R² | RMSE | MAE |
|----------|-------|-----|------|-----|
| LAIe | NDVI poly(3) | 0.695 | 0.690 | 0.479 |
| | S2BIOPHYS | 0.747 | 0.628 | 0.453 |
| FAPAR | NDVI poly(3) | 0.760 | 0.157 | 0.111 |
| | S2BIOPHYS | 0.747 | 0.161 | 0.113 |
| FCOVER | NDVI poly(3) | 0.754 | 0.156 | 0.112 |
| | S2BIOPHYS | 0.767 | 0.152 | 0.109 |
