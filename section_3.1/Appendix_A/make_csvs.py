import numpy as np 
import pandas as pd 
from glob import glob

def read_2012(path_to_dataset):
    '''
    Parameters:
    ----------
    path_to_dataset (string):
        Path to the CBECS dataset to read in

    Returns:
    --------
    data (pandas.DataFrame):
        All predictors
    '''

    data = pd.read_csv(path_to_dataset)

    # Drop imp variables
    for key in data.keys():
        if key[0] == 'Z':
            data.drop([key], inplace = True, axis = 1)

    return data

def read_in_year(year):
    '''
    Read in CBECS data for a given year
    Parameters: 
    -----------
    year: String
        year of dataset
    Returns:
    --------
    data: pd.Dataframe
        dataset
    '''
    path_to_files = "cbecs_data/{}/*".format(year)
    file_glob = glob(path_to_files)
    df_list = [pd.read_csv(micro_data) for micro_data in file_glob]
    data = df_list[0]

    for micro_data in df_list[1:]:

        # select the next data source
        duplicate_keys = [k for k in micro_data.keys() if k in data.keys()]

        # drop the currently included keys
        micro_data.drop(duplicate_keys, inplace = True, axis = 1)

        # add new predictors to the dataframe
        data = pd.concat((data, micro_data), axis = 1)

    # Fix up the header
    new_labels = {old_key:old_key.strip()[:-1] for old_key in data.keys()}
    data.rename(new_labels, inplace = True, axis = 1)

    return data

def make_cbecs():
    '''
    Make the complete CBECS dataset
    Returns:
    --------
    data_final: pd.Dataframe
        complete dataset
    '''

    df_list = [read_2012("cbecs_data/2012/2012_data.csv"), read_in_year(2003)]

    # Find the common keys in all datasets
    key_list = [set(df.keys()) for df in df_list]

    common_keys = key_list[0].intersection(*key_list)

    #common_keys = set([key.strip() for df in df_list for key in df.keys()])
    df_list = [df[common_keys] for df in df_list]

    data_final = pd.concat(df_list, axis = 0)
    data_final.reset_index(drop = True, inplace = True)

    return data_final

def make_rbecs():
    '''
    Make the complete RECS dataset
    Returns:
    --------
    data_final: pd.Dataframe
        complete dataset
    '''

    years = [2015, 2009]

    df_list = []

    for year in years:
        path = "rbecs_data/recs{}_public.csv".format(year)
        year_df = pd.read_csv(path)

        # change the fuel key
        if 'FUEL_HEAT' not in year_df.keys():
            year_df['FUEL_HEAT'] = year_df['FUELHEAT']

        # Drop imp variables
        for key in year_df.keys():
            if key[0] == 'Z':
                year_df.drop([key], inplace = True, axis = 1)

        df_list.append(year_df)

    # Find the common keys in all datasets
    key_list = [set(df.keys()) for df in df_list]
    common_keys = key_list[0].intersection(*key_list)

    #common_keys = set([key.strip() for df in df_list for key in df.keys()])
    df_list = [df[common_keys] for df in df_list]

    data_final = pd.concat(df_list, axis = 0)
    data_final.reset_index(drop = True, inplace = True)

    return data_final


if __name__ == "__main__":

    # Make the overall CBECS csv
    cbecs_data = make_cbecs()
    cbecs_data.to_csv("cbecs_2003_2012.csv")

    # Make the overall RBECS CSV
    rbecs_data = make_rbecs()
    rbecs_data.to_csv("rbecs_2005_2009_2015.csv")