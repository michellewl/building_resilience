# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd


def feature_engineer(df, time_col, key_col1, key_col2, seasonal=False, val_col=None, time_unit ='mon'):
    '''
    Add date characteristics to df, change wind direction to sine and cosine columns
    -------------------------------

    '''
    df['day'] = df[time_col].dt.day.astype(str)
    df['mon'] = df[time_col].dt.month.astype(str)
    df['weekday'] = df[time_col].dt.weekday.astype(str)
    df['year'] = df[time_col].dt.year
    df['date'] = df[time_col].dt.date
    df['hr'] = df[time_col].dt.hour
    df['night'] = df['hr'].apply(lambda x: 1 if (x < 6) | (x>=18) else 0)

    df['wind_dir_cos'] = np.cos(df['wind_direction'])
    df['wind_dir_sin'] = np.sin(df['wind_direction'])
    df = unique_key_from_two_cols(df, key_col1, key_col2)
    if (seasonal):
        df = add_seasonal_feat(df, 'ukey', val_col, time_unit = 'mon')


    return df



def unique_key_from_two_cols(df, col1, col2):
    '''
    
    '''
    df['ukey'] = df[[col1, col2]].apply(lambda x: str(x[0]) + chr(int(x[1]) + 65) + str(x[1]), axis =1)
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


def add_seasonal_feat(df, id_col, val_col, time_unit = 'mon'):
    '''
    Add explicit seasonal features for algo. where we expect to have historical data for future observations 
    --------------
    Parameters:
    df (pandas DataFrame)
    id_col (string): apply seasonal feature per id in id_col
    val_col (string): what column should we aggregate
    time_unit (string): deafult month, time unit for aggregation 
    ---------------
    Returns:
    pandas Dataframe with 2 additional columns 
    
    '''
    for idd in df[id_col].unique():
        monthly_count = np.array(df.loc[df[id_col] == idd, :].groupby(time_unit).count()[val_col]).reshape(-1)
        prev_mon_avg = np.array(df.loc[df[id_col] == idd, :].groupby(time_unit).mean()[val_col].shift().rolling(window=1).mean())
        prev_3_mon_avg = np.array(df.loc[df[id_col] == idd, :].groupby(time_unit).mean()[val_col].shift().rolling(window=3).mean())
        df.loc[df[id_col] == idd, 'prev_avg'] = np.repeat(prev_mon_avg, monthly_count)
        df.loc[df[id_col] == idd, 'prev_3_avg'] = np.repeat(prev_3_mon_avg, monthly_count)


    
    return df
