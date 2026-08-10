# S2BIOPHYS: Global High-Resolution Vegetation Maps

**Global 20m resolution maps of vegetation biophysical properties (LAIe, FAPAR, FCOVER) from Sentinel-2 data (2019-2025)**

**Quick Links:** [View Maps](https://ee-speckerfelix.projects.earthengine.app/view/global-trait-maps) | [Download Data](https://doi.org/10.5281/zenodo.19366930) | [Python Package](https://pypi.org/project/gee-biophys/) | [GEE Code Examples](https://code.earthengine.google.com/?scriptPath=users%2Fspeckerf%2Fopen-earth-public%3As2biophys-analysis-v3)

> **Status:** Under review - preprint available soon

## Table of Contents
- [Quick Start](#quick-start)
- [Data Access](#data-access)
- [Python Package](#python-package)
- [About This Project](#about-this-project)
- [Development Setup](#development-setup)
- [Repository Structure](#repository-structure)
- [Acknowledgments](#acknowledgments)

## Quick Start

### For Data Users
- **Interactive viewer:** [Global High-resolution Maps App](https://ee-speckerfelix.projects.earthengine.app/view/global-trait-maps)
- **Download ready-to-use data:** [Zenodo Repository](https://doi.org/10.5281/zenodo.19366930) (100m, 1000m resolution)
- **Use in Google Earth Engine:** [Code Examples](https://code.earthengine.google.com/?scriptPath=users%2Fspeckerf%2Fopen-earth-public%3As2biophys-analysis-v3)

### For Python Users
```bash
pip install gee-biophys
```
Generate custom spatiotemporal composites with simple configuration files.

### For Researchers
See [Development Setup](#development-setup) to reproduce model training and validation.

## About This Project

**Advancing Ecosystem Monitoring with Global High-Resolution Maps of Vegetation Biophysical Properties**

Human activities are rapidly transforming ecosystems worldwide, underscoring the urgent need for scalable approaches to monitor vegetation condition and recovery at scale. Public satellite missions such as Sentinel-2 have great potential for monitoring key vegetation biophysical properties because of their high spatial and temporal resolution. Radiative transfer models provide a physically based link between reflectance and vegetation structure and function, and radiative transfer-based inversion methods are well established for retrieving biophysical variables. However, existing operational products are either coarse in resolution, geographically restricted, or available only through on-demand processing workflows that require technical expertise. As a result, access to high-resolution, analysis-ready vegetation biophysical information remains limited, constraining large-scale ecosystem monitoring and restoration assessment. Here, we introduce S2BIOPHYS, the first global operational product providing precomputed, annual maps at high resolution (20 m) for three key vegetation biophysical properties: effective leaf area index (LAIe), fraction of absorbed photosynthetically active radiation (FAPAR) and fractional vegetation cover (FCOVER). To generate these products, we systematically extend radiative transfer-based inversion approaches by coupling simulation-driven model training with empirically constrained parameter optimization using an extensive in-situ reference dataset. Annual composites for 2019-2025 are derived from peak growing-season Sentinel-2 observations and include per-pixel mean values, calibrated uncertainty estimates and observation counts. Validation against more than 11,000 ground reference measurements, evaluated using spatially independent cross-validation, demonstrates strong performance relative to existing retrieval approaches. The resulting ~20 TB global dataset, together with an accompanying Python package enabling sub-annual retrievals, supports downstream applications in biodiversity assessment, ecosystem restoration monitoring, and global reporting frameworks.

## Disclaimer

The output of this work is currently under revision and is thus not yet peer-reviewed. Preprint available soon. 

## Data Access

### Download Ready-to-Use Data
| Resolution | Format | Source | Projection |
|------------|--------|--------|------------|
| 100m / 1000m | COG | [Zenodo](https://doi.org/10.5281/zenodo.19366930) | EPSG:4326 |

### Interactive Access
- **Web App**: [Global High-resolution Maps of Biophysical Vegetation Properties](https://ee-speckerfelix.projects.earthengine.app/view/global-trait-maps)
- **20m Resolution**: Available through Google Earth Engine

### Google Earth Engine Assets
```js
// Asset collections (20m and 100m resolution)
ee.ImageCollection('projects/ee-speckerfelix/assets/open-earth/[fapar,laie,fcover]_predictions-mlp_[100m,20m]_v03')
```

**Code Examples:**

- [Visualization Code](https://code.earthengine.google.com/?scriptPath=users%2Fspeckerf%2Fopen-earth-public%3As2biophys-visualize-v3)
- [Analysis Examples](https://code.earthengine.google.com/?scriptPath=users%2Fspeckerf%2Fopen-earth-public%3As2biophys-analysis-v3)

<details>
<summary>Example: Load yearly image in GEE</summary>

```js
var resolution = '100m' // 100m / 20m

// Trait-specific scaling factors
var scalingFactors = {
  'fapar': 10000,
  'fcover': 10000,
  'laie': 1000
};

// Function to retrieve the image collection based on trait, version, and year
function get_yearly_image(trait, year) {
  var collectionPath = 'projects/ee-speckerfelix/assets/open-earth/' + trait + '_predictions-mlp_' + resolution + '_v03';
  var image = ee.ImageCollection(collectionPath).filterDate(year + '-01-01', year + '-12-31').mosaic();
  var mean_band = trait + '_mean'
  var std_band = trait + '_stdDev'
  var count_band = trait + '_count'
  image = image.select([mean_band, std_band]).divide(scalingFactors[trait]).addBands(image.select([count_band]))
  return image;
}

var laie_2019 = get_yearly_image('laie', 2019);
```
</details>

## Python Package

### gee-biophys
The [`gee-biophys`](https://pypi.org/project/gee-biophys/) Python package enables users to generate custom spatiotemporal composites of vegetation biophysical properties. It implements both S2BIOPHYS and SL2P algorithms, allowing users to define time intervals, spatial extents, and resolution through a simple configuration file.

**Features:**
- Reproducible and scalable generation of biophysical maps (LAIe, FAPAR, FCOVER)
- Access to full uncertainty components at native resolution
- Simple configuration-based workflow

**Installation:**
```bash
pip install gee-biophys
```

## Development Setup

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone git@github.com:speckerf/world-reforestation-monitor.git
   cd world-reforestation-monitor
   ```

2. **Create environment**
   ```bash
   conda env create -f environment.yml
   conda activate world-reforestation-monitor
   ```

3. **Download data**
   - Download `data.tar.gz` from [Zenodo](https://doi.org/10.5281/zenodo.15052996)
   - Extract to project root: `tar -xzf data.tar.gz`

### Optional: Model Training Setup

<details>
<summary>Click to expand advanced setup for model training</summary>

**Install Modified PROSAIL R Package:**
```r
# In RStudio
library(devtools)
devtools::install_github("speckerf/prosail")
```

**Setup OPTUNA Database:**
```bash
# Install MySQL (tested with version: MySQL 8.4)
brew install mysql@8.4

# Start MySQL
brew services start mysql@8.4

# Create database
mysql -u root -e "CREATE DATABASE oemc;"
```

Update `config/train_pipeline.yaml` with database connection details.

**Monitor Training:**
```bash
python -m train_pipeline.optunaTraining
optuna-dashboard mysql://root@localhost/oemc
```
</details>

### Running Scripts

Scripts should be run as modules from the root directory:
```bash
# Example
python -m train_pipeline.optunaTraining
```

**VS Code Debug Configuration:**
```json
{
    "version": "0.2.0",
    "configurations": [{
        "name": "Python: Debug Current Script as Module",
        "type": "debugpy",
        "request": "launch",
        "module": "train_pipeline.optunaTraining",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}"
    }]
}
```


## Repository Structure

### Core Modules

| Module | Purpose | Key Scripts |
|--------|---------|-------------|
| **validation_pipeline** | Process GROUNDED-EO validation data, export Sentinel-2 reflectances | Data processing and validation |
| **train_pipeline** | Model training with Optuna hyperparameter optimization | `optunaTraining.py`, `finalTraining.py` |
| **gee_pipeline** | Google Earth Engine server-side computations and exports | `srcGlobal.py`, `srcOrbits.py` |
| **ee_translator** | Helper classes for sklearn to GEE translation | Model conversion utilities |

### Supporting Directories

| Directory | Contents |
|-----------|----------|
| **config/** | Configuration YAML files for all modules |
| **rtm_pipeline_R/** | Command-line scripts for PROSAIL forward model |
| **rtm_pipeline_python/** | Python classes for PROSAIL input generation |
| **data/** | Input datasets (downloaded separately) |
| **auth/** | Authentication credentials (not in git) |


## Citation

> **Note:** This work is currently under peer review. Please check for updates and cite the published version when available.

## License

This project uses a dual licensing approach:

- **Source Code**: Licensed under the terms specified in [LICENSE.txt](LICENSE.txt)
- **Data Products**: All produced vegetation maps are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  

## Acknowledgments

The Open-Earth-Monitor Cyberinfrastructure (OEMC) project has received funding from the European
Union’s Horizon Europe research and innovation programme under grant agreement No.
101059548.
