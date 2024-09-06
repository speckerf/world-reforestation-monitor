# Global vegetation trait maps

**Setup:**
- create environment from environment.yaml
- To export: ensure that both gcloud and earthengine-api are installed and authenticated. 
- Data repository is tracked using data version control:
    - Data is stored in a private bucket on google cloud storage / access needs to be manually granted. 
    - After that: pull the data using: 'dvc pull'
- usually, all files need to be run as a module from the root of the directory:
    - e.g. python -m training_pipeline.optunaTraining
    - if used in vscode in debug-mode: add the following to your launch.json file:
        - {
                "version": "0.2.0",
                "configurations": [
                    {
                        "name": "Python: Debug Current Script as Module",
                        "type": "debugpy",
                        "request": "launch",
                        "module": "train_pipeline.finalTraining",
                        "console": "integratedTerminal",
                        "justMyCode": true,
                        "args": [],
                        "cwd": "${workspaceFolder}"
                    }
                ]
            }
- Sometimes, scripts need to be run in a specific order, which might not be obvious - I need to clean the repo in that regard

**Repository Structure**
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

- phenology_pipeline: 
    - contains code to obtain the growing season period for every ecoregion on earth. 
    - automated growing season determinaiton based oon historical EVI product
    - then applies several manual adjustements necessary to improve the consistency of the resulting maps

- config:
    - holds a configuration yaml file for most modules

- rtm_pipeline_R:
    - contains a command line script that calls a modified version of the prosail package. 

- rtm_pipeline_python:
    - contains a class for generating the inputs for the prosail forward model. 
    - inputs are created based on some predefined parameter ranges: see config/rtm_simulator/*.yaml


**Data:**
- data is stored in data folder and tracked using dvc (data version control). This means the data folder actually only contains pointers to the actual version controlled data stored as a hash table in .dvc/cache/files
- the remote repository for the data is at: gs://felixspecker/open-earth/world-reforestation-monitor-remote
- if you change update any data in the data repo and want to track that:
    - dvc add data
    - git add data.dvc
    - git commit
    - dvc push

