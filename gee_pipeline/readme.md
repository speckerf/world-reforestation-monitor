# Model Prediction

![alt text](../figures_python/plots/figure_1.png)

This repository contains scripts to predict LAI, FAPAR and FCOVER globally. Uses pretrained models that were the result of the module `train_pipeline`. 

## Overview
The `gee_pipeline` module facitlitates the global export of LAI/FAPAR and FCOVER maps at user-defined spatial resolutions and years, as specified in the `config/gee_pipeline.yaml` configuration file. The pipeline includes various utility scripts for processing Sentinel-2 data, cloud masking, and trait prediction using machine learning models converted for Google Earth Engine (GEE).

### `srcGlobal.py`
This is the main entry point of the pipeline. It initiates the global export process for vegetation trait predictions based on. Make sure to run all scripts as modules, e.g.:
```bash
python -m gee_pipeline.srcGlobal
```

### `srcOrbits.py`
This module determines relevant **Sentinel-2 orbits** for each **MGRS tile**. Due to high orbital overlap in northern latitudes, not all orbits are needed to effectively capture the entire MGRS tile extent. This script preprocesses and retains only the most relevant orbits per tile, optimizing processing efficiency.

### `utilsTiles.py`
Processes Sentinel-2 acquisitions for a **given Sentinel-2 tile**:
1. **Filters all acquisitions** within a specified year.
2. **Filters for maximum cloud cover** based on the settings in `config/gee_pipeline.yaml`.
3. **Performs pixel-based cloud masking** (see `utilsCloudfree.py`).
4. **Calculates tile-level NDVI**, ensuring that non-natural land cover types (e.g., agricultural areas) are excluded to focus on natural vegetation phenology.
5. **Selects the top 8 acquisitions** based on a function favoring **high NDVI** and **low cloud cover**.

### `utilsCloudfree.py`
Performs **cloud masking** by:
- Pairing each selected Sentinel-2 scene with the **Cloudscore+ collection**.
- Effectively masking out clouds on a **per-pixel basis**.

### `utilsPredict.py`
Handles machine learning predictions for vegetation traits:
- Loads **pre-trained models**.
- Converts prediction functionality to **Google Earth Engine (GEE) code**.
- Uses the `ee_translater` submodule, specifically:
  - `ee_mlp_regressor.py`: Implements **MLP regression models** for trait prediction.
  - `ee_standard_scaler.py`: Applies **standardization** to input features in GEE.

### `utilsOOD.py`
Provides **out-of-distribution (OOD) filtering**:
- Masks out reflectance values that fall **outside the range of simulated reflectance values**.
- Performs this filtering **on a per-band basis** rather than multivariate filtering.

## Configuration
All pipeline parameters (e.g., year, trait, resolution, cloud filtering thresholds) are stored in:
```yaml
config/gee_pipeline.yaml
```
Ensure that this file is properly configured before running the pipeline.
