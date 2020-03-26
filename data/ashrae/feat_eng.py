# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd


def feature_engineer(df, time_col):
    '''
    Add date characteristics to df, change wind direction to sine and cosine columns
    -------------------------------

    '''
    df['day'] = df[time_col].dt.day.astype(str)
    df['mon'] = df[time_col].dt.month.astype(str)
    df['weekday'] = df[time_col].dt.weekday.astype(str)
    df['year'] = df[time_col].dt.year
    df['date'] = df[time_col].dt.date

    df['wind_dir_cos'] = np.cos(df['wind_direction'])
    df['wind_dir_sin'] = np.sin(df['wind_direction'])

    return df


def one_hot(df, columns):
    '''
    one hot encode columns and concat to original df
    ----------
    Parameters: 
    df (pandas dataframe)
    columns (list of strings): columns to one hot encode

    -------
    Return:
     One pandas dataframe with the one hot encoded columns 

    '''
    return pd.concat((df, pd.get_dummies(df[columns])), axis=1)
