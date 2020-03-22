import numpy as np


def cv_metric(obs, pred):
    '''
    normalizes the prediction error by the average energy consumption
    and provides a unitless measure that is more convenient for comparison purposes
    --------
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
    Accuracy measure based on percentage (or relative) errors.
    Percentage error between 0% and 100%
    not symmetric.
    e.g. The difference between 0 and 1 isn't the same as 100 and 101!
    ---------
    obs (array of floats)
    pred (array of floats)
    -------
   Returns:
   SMAPE (float)

    '''
    n = len(obs)
    return np.sum(np.abs(pred - obs) / ((np.abs(obs) + np.abs(pred)))) * 100 / n