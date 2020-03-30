# -*- coding: UTF-8 -*-
import pandas as pd

def read_n_join_data(path):
    '''
    Read buildings data and join with metadata
    -----------------
    Parameters:
    path (str): where the buildings data can be found

    -----------------
    Returns:
    ener_w_met_n_wea (pandas DataFrame): energy & weather data
    meta (pandas DataFrame): Building characteristics

    '''
    energy_tr = pd.read_csv(path + '/train.csv')
    meta = pd.read_csv(path + '/building_metadata.csv')
    energy_w_meta = pd.merge(energy_tr, meta, how='left', on="building_id")
    weather = pd.read_csv(path + '/weather_train.csv')
    ener_w_met_n_wea = pd.merge(energy_w_meta, weather, how="left", on=[
                                "site_id", "timestamp"])
    return ener_w_met_n_wea, meta

