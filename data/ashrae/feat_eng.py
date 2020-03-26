# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

def feature_engineer(df, time_col):
    '''
    Add date characteristics to df, change wind direction to sine and cosine columns
    -------------------------------
    
    '''
    df['day'] = df[time_col].dt.day.astype(str)
    df['mon'] = df[time_col].dt.month.astype(str)
    df['weekday'] = df[time_col].dt.weekday.astype(str)
    df['year'] = df[time_col].dt.year
    
    df['wind_dir_cos'] = np.cos(df['wind_direction'])
    df['wind_dir_sin'] = np.sin(df['wind_direction'])


    
    return df