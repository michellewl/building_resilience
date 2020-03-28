# -*- coding: UTF-8 -*-
import pandas as pd 

def merge_df_on_id(df, df_m, column_name, typ="left"):
    '''
    merge two pandas df
    ----------
    Parameters: 
    df (pandas dataframe): the left df
    df_m (pandas dataframe): right df
    column_name (string): column to merge on
    typ (string): left (default), right, inner
    -------
    Return:
     One merged pandas dataframe 

    '''
    return pd.merge(df, df_m, how=typ, on=column_name)


def one_hot(df, columns):
    '''
    merge two pandas df
    ----------
    Parameters: 
    df (pandas dataframe)
    columns (list of strings): columns to one hot encode
    column_name (string): column to merge on
    typ (string): left (default), right, inner

    -------
    Return:
     One pandas dataframe with the one hot encoded columns 

    '''
    return pd.concat((df, pd.get_dummies(df[columns])), axis=1)
