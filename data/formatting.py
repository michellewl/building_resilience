# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np


def formatting(df, time_col, cols_to_str):
    '''
    Change columns' format, time col to datetime, columns that are one hot encoded subsequently to str
    --------------------------
    Parameters:
    df (pandas DataFrame)
    time_col (str): column name for time data
    cols_to_str (list of strings): columns to later one hot encode
    
    -------------------------
    Returns:
    formatted df (pandas DataFrame)
    
    '''
    df[time_col] = pd.to_datetime(df[time_col])
    df = format_col_to_str(cols_to_str)
    df['meter_per_sqft'] = df['meter_reading'] / df['square_feet']


    return df