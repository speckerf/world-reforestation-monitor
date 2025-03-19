# Code and Data repository

## Title: Advancing Ecosystem Monitoring: Global High-resolution Maps of Biophysical Vegetation Properties

Restoration projects are crucial for recovering ecosystems and conserving biodiversity but
monitoring progress at a large scale poses significant challenges. Public satellite missions like
Sentinel-2 have great potential to transform ecosystem monitoring due to their high spatial and
temporal resolution if they can be reliably linked to ecosystem characteristics. Here, we present
the first high-resolution global maps for three key biophysical vegetation properties on an annual
basis, including leaf area index (LAI), fraction of absorbed photosynthetically active radiation
(FAPAR), and fractional vegetation cover (FCOVER). We utilize a hybrid inversion approach of
the mechanistic radiative transfer model PROSAIL to directly estimate biophysical variables
from multispectral Sentinel-2 images, considering multiple observations during the peak growing
season. All predictions are aggregated into mean values, standard deviations, and the number
of observations taken during this period. The maps are available at 20m, 100m, and 1000m
spatial resolution for the years 2019 to 2024, totalling approximately 20TB of analysis-ready
data, and are validated using in-situ ground-based observations. The high temporal and spatial
resolution of these maps provides new opportunities for biodiversity and ecosystem monitoring,
enabling more effective assessments of restoration efforts and contributing to the development
of standardized global monitoring frameworks.

## Global LAI/FAPAR/FCOVER maps - Data availability

Maps are available at resolutions: 

**1000m**
- **Source**: GEE (see below) or [Zenodo](TODO)
- **Projection**: Single COGs in `EPSG:4325`


**100m**
- **Source**: GEE (see below) or multi-bucket deposition at [Zenodo](TODO)
- **Projection**: Single COGs in `EPSG:4325`

**20m**: 
- **App**: [Global Trait Maps](https://ee-speckerfelix.projects.earthengine.app/view/global-trait-maps)
- **Assets**: 
```js
ee.ImageCollection(`projects/ee-speckerfelix/assets/open-earth/[fapar, lai, fcover]_predictions-mlp_[1000m, 100m, 20m]_v01`)
```
- **Projection**: Local UTM Projection (native Sentinel-2 projection, requires mosaicing)
- **Visualization Code** [View in GEE Code Editor](https://code.earthengine.google.com/7207cd15a5cc312ac816dc76cd60b450)
- **Example: Mosaicking / Scaling / Filtering Code** [Open in GEE Code Editor](https://code.earthengine.google.com/22fc7da25a4dbe758988cbee9afcf763)
```js
    var resolution = '100m' // 1000m / 100m / 20m

    // Trait-specific scaling factors
    var scalingFactors = {
    'fapar': 10000,
    'fcover': 10000,
    'lai': 1000
    };

    // Function to retrieve the image collection based on trait, version, and year
    function get_yearly_image(trait, year) {
        var collectionPath = 'projects/ee-speckerfelix/assets/open-earth/' + trait + '_predictions-mlp_' + resolution + '_v01';
        var image = ee.ImageCollection(collectionPath).filterDate(year + '-01-01', year + '-12-31').mosaic();
        var mean_band = trait + '_mean'
        var std_band = trait + '_stdDev'
        var count_band = trait + '_count'
        image = image.select([mean_band, std_band]).divide(scalingFactors[trait]).addBands(image.select([count_band]))
        return image;
    }

    var lai_2019 = get_yearly_image('lai', 2019)`
```

## Setup Environment

To setup to environment, please follow these steps:

1. Clone this repository: `git clone git@github.com:speckerf/world-reforestation-monitor.git`
2. Create virtual environment from envirnoment.yml file: (we recommend to use conda or mamba):
    - `conda env create -f environment.yml`

3. Download `data` repository. 
    - The folder `data` is not directly stored in the GitHub repository due to the large storage size. However, it is tracked using data-version-control. The obtain a snapshot of the data folder, download the file `data.tar.gz` from zenodo and uncompress locally. Move it in the root directory of the project, such that all relative paths work. 
    - Alternatively, the remote location of the dev-tracked data folder is stored in a private Google Cloud Storage bucket / but access needs to be manually granted. 

4. \[Optional: Only required for model training\]: 
    1. Install modified `prosail` R package (forked and modified from https://github.com/jbferet/prosail)
        - Specifically, the fork modifies the function `prosail::Generate_LUT_PROSAIL` and adds the possibility of using custom soil spectra. 
        - open `RStudio`
        - execute the following two lines:
            - `library(devtools)`
            - `devtools::install_github("speckerf/prosail")`
    2. Setup OPTUNA Hyperparameter tuning framework
        - Install `mysql` (or other SQL database)
        - Create a database: e.g. `CREATE DATABASE oemc;`
        - Make sure that the configuration `config/train_pipeline.yaml` contains the right database storage location. 
        - run `python -m train_pipeline.optunaTraining` and observe the training progress using `optuna-dashboard` (e.g. optuna-dashboard mysql://root@localhost/oemc)

5. Running Scripts: 
    - Most scripts need to be run as modules from the root directory, e.g.: 
        ```
        python -m train_pipeline.optunaTraining
        ```
    - You can setup a `launch.json` in `/.vscode/` in order to run and debug a specific file as a module, see e.g.:
                ```
                    {
                        "version": "0.2.0",
                            "configurations": [
                                {
                                    "name": "Python: Debug Current Script as Module",
                                    "type": "debugpy",
                                    "request": "launch",
                                    "module": "train_pipeline.optunaTraining",
                                    "console": "integratedTerminal",
                                    "justMyCode": true,
                                    "args": [],
                                    "cwd": "${workspaceFolder}"
                                }
                            ]
                    }
                ```
    - Please refer to the module-specific readme.md files in order to get more information on how to execute the given code. 


## Repository Structure
The repo is structure in different modules:
- validation_pipeline:
    - processes the downloaded GBOV or NEON validation data
    - export the sentinel-2 reflectances for every validation data point
    - downloads the exports and merges them locally

- train_pipeline:
    - train models using optuna framework for hyperparameter optimisation
    - optunaTraining.py: starts an optuna experiment 
    - finalTraining.py: after optuna experiments finished, rerun the best models and save them to disk

- gee_pipeline: 
    - implements the code to translate the best models to perform GEE server-side computations and starts the export for the trait maps
    - gee_pipeline.srcGlobal starts global export of trait maps at specified resolution : but export can also be limited to specific ecoregions for testing. 
    
- ee_translator: 
    - stores some helper classes that translate the functionalities of sklearn-type classes, to GEE server side operations on type: ee.Image

- config:
    - holds a configuration yaml file for most modules

- rtm_pipeline_R:
    - contains a command line script that calls a modified version of the prosail package. 

- rtm_pipeline_python:
    - contains a class for generating the inputs for the prosail forward model. 
    - inputs are created based on some predefined parameter ranges: see config/rtm_simulator/*.yaml
