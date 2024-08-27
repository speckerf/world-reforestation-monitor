# Phenology Pipeline
For every ecoregion, we extract the phenological peak of the growing season dates. 

- average_pheno_timeseries.py
    - takes the modis product, and calculates the average day of the year for different phenology indices. Export this map in GEE
- export_ecoregion_pheno.py
    - reads the exported timeseries average pixel maps, and performs spatial reductions to obtain mean estimates per ecoregion. Single export per ecoregion. 
- merge_ecoregion_pheno.py
    - Merges all the ecoregion level exports, and does some manual cleaning, like e.g. 
        - if numdays < 45: extend growing period to 45 days at both ends. 
        - if the ecosystem doesnt seem to have very distinct growing season (moist broadleef rain forest), take full 365 days:
            - defined as: amplitude (TODO: see code )


