# -*- coding: UTF-8 -*-
import numpy as np


def coef_of_var_metric(obs, pred):
    '''
    obs (array of floats)
    pred (array of floats)
    -------

   Returns:
   Coefficient of variation (float)
    
    '''
    n = len(obs)
    return (np.sqrt(np.sum(np.square(obs - pred)) / n) / np.mean(obs)) * 100


def rmse(obs, pred):
    '''
    obs (array of floats)
    pred (array of floats)
    -------

   Returns:
   Root mean squared error (float)
    
    '''
	n = len(obs)
	return np.sqrt(np.sum(np.square(obs - pred)) / n)


def smape(obs, pred):
    '''
    obs (array of floats)
    pred (array of floats)
    -------

   Returns:
   SMAPE (float)
    
    '''
	n = len(obs)
	return np.sum(np.abs(pred - obs) /  ((np.abs(obs) + np.abs(pred)) / 2)) * 100 / n
