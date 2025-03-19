# Perparation of Validation Data

![alt text](../figures_python/plots/figure_1.png)

This repository contains scripts for generating the validation data. Performs filtering on the LAI, FAPAR and FCOVER reference measurements obtained from GBOV. Also performs spatio-temporal overlay to obtain

**Workflow Overview**
1. First run the scripts `src_lai.py`, `src_fapar.py`, `src_fcover.py`. This mainly performs two tasks:
    - Load, filter and clean the GBOV reference measurements for the variable of interest. 
    - Uses GEE to find closest cloud-free Sentinel-2 observation and export the results to Google Cloud Storage. 
        -  Exports are performed per site. 
    
2. Then, download the exported results from Google Cloud Storage and merge to obtain a single csv file. 
    - Run `merge_fapar_sites.py`, `merge_fcover_sites.py`, `merge_lai_sites.py`

3. Resulting validation datasets are stored at:
    - FCOVER: `data/validation_pipeline/output/fcover/merged_fcover_COPERNICUS_GBOV_RM4_20240816101306.csv`
    - LAI: `data/validation_pipeline/output/lai/merged_lai_COPERNICUS_GBOV_RM6,7_20240620120826.csv`
    - FAPAR: `data/validation_pipeline/output/fapar/merged_FIPAR_COPERNICUS_GBOV_RM6,7_20240620120826.csv`

    