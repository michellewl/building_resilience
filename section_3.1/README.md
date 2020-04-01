# Section 3.1
## Annual energy demand modelling
### Author: Anna Vaughan
### Last updated: 1/4/2020

# Section 3.1
Notebooks for the model training and testing described in Section 3.1. 

### Model training
commercial_models.ipynb - heating and cooling model training for commercial buildings
residential_models.ipynb - heating and cooling model training for residential buildings

### Model interpretation
commercial_model_interpretation.ipynb - SHAP interpretation of trained commercial models
residential_model_interpretation.ipynb - SHAP interpretation of trained residential models

### Further scripts
data_clean_new.py - functions to read in training data
model_builder_new.py - functions for grid search and plotting

# Appendices

## Appendix A
Further scripts are contained in the appendix folders

make_csvs.py - make the raw RBECS/CBECS csvs
summarise_data.py - print data statistics

## Appendix C

Initial scripts for creating the global maps described in Appendix C

buildings.py - Class for open street map building objects
grid_cell.py - Class for object representing one grid cell in global maps
make_building_map.py - Make present day global energy usage maps for 2018 using open street map data
make_lim_predictor_datasets.py - subset features to those included in open street maps
tag_dicts.py - building type conversion from open street map format to CBECS/RECS format

### ERA_HDD_CDD

Scripts for calculating HDD/CDD from ERA Reanalysis T2 data
calc_hdd.py - retrieve HDD/CDD from ERA T2 data netcdf
get_ERA_data.sh - load required modules and get ERA5 T2 data from the Australian National Computational Infrastructure
get_T2_2018.py - Get the 2018 mean daily T2 over a specified area


