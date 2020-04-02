# -*- coding: UTF-8 -*-
import pandas as pd
import formatting as fm
import feat_eng as fe


def cleanup(df, ones_to_hot, ones_to_drop):
    '''
    full cleanup of building dataset ASHARE
    -----------------
    Parameters:
    
    df (pandas DataFrame): the time column is specified by column 'timestamp'
    ones_to_hot (list of strings): columns to encode
    ones_to_drop (list of strings): columns to drop from df
    -----------------
    Returns:
    df (pandas DataFrame): cleaned.
    
    '''
    df = fm.formatting(df, 'timestamp')
    df = fe.feature_engineer(df, 'timestamp')
    df = fe.one_hot(df, ones_to_hot)
    df = drop_redundant(df, ones_to_drop)
    # df = drop_outliers() ## still needs to be worked on
    df = missing_vals(df)
    return df


def drop_redundant(df, ones_to_drop):
    '''
    Drop Redundant columns, plus rows with meter other then 0 meter.
    ----------------
    Parameters:
    
    df (pandas DataFrame)
    ones_to_drop (list of strings): columns to drop from df
    
    ----------------
    Returns:
    df (pandas DataFrame): excluding columns that were dropped
    
    '''
    df = df.loc[(df['meter'] == 0) & (df['primary_use'] != 'Parking'), :]
    df.drop(ones_to_drop, axis = 1, inplace = True)
    
    
    return df


def missing_vals(df):
    '''
    Drop columns with fewer than 30% data
    --------------------
    Parameters:
    df (pandas DataFrame)
    --------------
    Returns:
    df (pandas DataFrame)
    
    '''
    for col in df.columns:
        if ((df[col].count() / df.shape[0]) < 0.3):
            df.drop(col, axis = 1, inplace=True)
    
    return df 