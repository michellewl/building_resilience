import numpy as np 
import pandas as pd 
from glob import glob

def drop_nan_ps(data_frame):
    '''
    Delete features where data is missing for >90% of buildings
    Parameters:
    -----------
    data_frame: pd.Dataframe
        feature data frame
    Returns:
    --------
    data_frame: pd.Dataframe
        cleaned data frame
    '''
    # Delete any predictor without data for >90% of buildings
    nan_sums = data_frame.isna().sum()

    for key in data_frame.keys():
        if nan_sums[key]/len(data_frame) > 0.1:
            data_frame.drop([key], inplace = True, axis = 1)

    return data_frame

def filter_data(targets, data, filter_key, file_id, filter_val = 1):
    '''
    Filter data according to a particular key value and save
    Parameters:
    -----------
    targets: pd.Series
        targets
    data: pd.Dataframe
        features
    filter_key: String
        key to filter over
    file_id: String
        identifier for saving files
    '''

    filtered_data = data[data[filter_key]==filter_val]
    targets = targets[filtered_data.index]
    predictors = filtered_data[~targets.isna()]
    targets = targets[~targets.isna()]

    # Drop features with >90% missing data
    predictors = drop_nan_ps(predictors)

    # Replace nans with -1
    for key in predictors.keys():
        predictors[key][predictors[key].isna()] = -1

    # Save files
    predictors.to_pickle(file_id+"_all_predictors.pkl")
    targets.to_pickle(file_id+"_use_targets.pkl")

def clean_cbecs_data():
    '''
    Clean the CBECS data and output in a format suitable for model training
    '''

    # Read in CBECS data
    cbecs_data = pd.read_csv('cbecs_2003_2012.csv')

    # Convert to numeric type
    for key in cbecs_data.keys():
        cbecs_data[key] = pd.to_numeric(cbecs_data[key], errors = "coerce")

    # Drop unnecessary features
    cbecs_data.drop(['Unnamed: 0'], inplace = True, axis = 1)
    nn_predictors = ['SQFTC', 'REGION', 'CENDIV', 'YRCONC', 'PBAPLUS', 'NOCCAT']
    cbecs_data.drop([p for p in nn_predictors if p in cbecs_data.keys()], axis = 1, inplace = True)
    cbecs_data['FREESTN'][cbecs_data['FREESTN'].isna()] = 0

    # Fix the label formatting
    new_labels = {old_key:old_key.strip() for old_key in cbecs_data.keys()}
    cbecs_data.rename(new_labels, inplace = True, axis = 1)

    # Get relevant energy use targets
    elec_heating_use = cbecs_data['ELHTBTU']
    elec_cooling_use = cbecs_data['ELCLBTU']
    
    # Drop all other energy use features
    energy_keys = [key for key in cbecs_data.keys() if key[-3:]=="BTU"]
    cbecs_data.drop(energy_keys, inplace = True, axis = 1)

    # Electricity cooling use 
    filter_data(elec_cooling_use, cbecs_data, 'ELCOOL', 'elec_cooling')

    # Electricity heating use
    filter_data(elec_heating_use, cbecs_data, 'ELHT1', 'elec_heating')

def clean_rbecs_data():
    '''
    Clean the RECS data and output in a format suitable for model training
    '''

    # Read in raw RBECS data
    rbecs_data = pd.read_csv('rbecs_2005_2009_2015.csv')

    # Convert to numeric type
    for key in rbecs_data.keys():
        rbecs_data[key] = pd.to_numeric(rbecs_data[key], errors = "coerce")
    
    # Drop unnecessary features
    drop_p = ['DOEID', 'REGIONC', 'DIVISION', 'METROMICRO', 'Unnamed: 0']
    rbecs_data.drop(drop_p, axis = 1, inplace = True)

    # Clean the 'STORIES' feature
    rbecs_data['STORIES'][rbecs_data['STORIES'] == 40] = 1.5
    rbecs_data['STORIES'][rbecs_data['STORIES'] == 10] = 1
    rbecs_data['STORIES'][rbecs_data['STORIES'] == 20] = 2
    rbecs_data['STORIES'][rbecs_data['STORIES'] == 31] = 3
    rbecs_data['STORIES'][rbecs_data['STORIES'] == 32] = 4
    rbecs_data['STORIES'][rbecs_data['STORIES'] == -2] = -2

    # Get relevant energy use targets
    elec_heating_use = rbecs_data['KWHSPH']
    elec_cooling_use = rbecs_data['KWHCOL']

    # Drop all other energy use features
    drop_keys = [key 
            for key in rbecs_data.keys() 
                if key.startswith(('BTU', 'DOL', 'CUFEE', 'GALL', 'TOTAL', 'KW'))]

    rbecs_data.drop(drop_keys, inplace = True, axis = 1)

    # Electricity cooling use 
    filter_data(elec_cooling_use, rbecs_data, 'ELCOOL', 'res_elec_cooling')

    # Electricity heating use
    filter_data(elec_heating_use, rbecs_data, 'FUEL_HEAT', 'res_elec_heating', filter_val=5)