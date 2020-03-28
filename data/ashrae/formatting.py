# -*- coding: UTF-8 -*-
import pandas as pd


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
    df[cols_to_str] = format_col_to_str(df, cols_to_str)
    df['meter_per_sqft'] = df['meter_reading'] / df['square_feet']

    return df


def format_col_to_str(df, cols):
    '''
    Convert multiple columns to str format
    ---------------------
    Parameters:
    df (pandas DataFrame)
    cols (list of strings): cols to convert
    '''

    return df[cols].astype(str)
