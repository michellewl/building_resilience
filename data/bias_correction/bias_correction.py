# -*- coding: UTF-8 -*-
import dataprocessing as dp
import pandas as pd
import xarray as xr
import numpy as np


def remove_leaps_xarray(xr):
    '''
    Remove all 29-02 days
    ---------------
    Parameters: 
    xr (Xarray)
    ---------------
    Returns: 
    xr (Xarray) without 29-02 days
    '''
    return xr.where(((xr.time.dt.day != 29) | (xr.time.dt.month != 2)), drop=True)


def match_calendar(obs, observations=True):
    '''
    Matching between a 360 day calendar and regular calendaric year
    ----------------------
    Parameters:
    obs (xarray): must have a time variable
    observations (boolean): deafult (True), correct the observations or correct climate models
    ---------------------
    Returns:
    xarray corrected to have same calendar for climate model and observations

    '''
    if (observations):
        obs = obs.where(obs.time.dt.day != 31, drop=True)
        obs = remove_leaps_xarray(obs)
    else:
        obs = obs.where(((obs.time.dt.day != 30) | (
            obs.time.dt.month != 2)), drop=True)
        obs = remove_leaps_xarray(obs)
    return obs


def construct_ecdf(np_array):
    '''
    Construct an empirical cumulative distribution function
    ----------------
    Parameters:
    np_array (numpy array)
    _______________
    Returns:
    sorted_values (numpy array): the input array sorted
    pr_smaller_x (numpy array): the prob. to be smaller equal to the value in sorted_values

    '''
    sorted_values = np.sort(np_array)
    pr_smaller_x = []
    cum_val = 0
    for item in range(len(sorted_values)):
        if (item > 0):
            if (sorted_values[item] != sorted_values[item - 1]):
                cum_val = pr_smaller_x[item - 1]

        pr_smaller_x.append(
            ((sorted_values == sorted_values[item]).sum() / len(sorted_values)) + cum_val)

    return sorted_values, pr_smaller_x


def ecdf_val(val, ecdf_vals, ecdf_pr):
    '''
    get probability for a certain value given empirical cumulative distribution function
    ----------------- 
    Parameters: 
    val (float): the value for which p(x<= val) is wanted
    ecdf_vals (numpy array): the values from given empirical distribution
    ecdf_pr (numpy array): the probabilities from given empirical distribution
    -----------------
    Returns: (float), p(x<= val)
    '''

    if all(val >= ecdf_vals):
        return 1
    idx = np.where(ecdf_vals >= val)[0]
    return ecdf_pr[idx.astype(int)[0]]


def inverse_ecdf_pr(pr, ecdf_vals, ecdf_pr):
    '''
    Get the value associated with a certain probability value from the empirical cumulative distribution function
    ------------------
    Parameters: 
    pr (float): the probability for which p(x<= val) = pr is wanted
    ecdf_vals (numpy array): the values from given empirical distribution
    ecdf_pr (numpy array): the probabilities from given empirical distribution
    ------------------
    Returns:
    val (float) such that p(x<= val) = pr
    '''
    if (all(pr >= ecdf_pr)):
        return np.max(ecdf_vals)
    idx = np.where(ecdf_pr >= pr)[0]
    return ecdf_vals[idx.astype(int)[0]]


def detrend_seasonality(np_array, window=360):
    '''
    Remove seasonal trend by subtracting from each day, same day in previous year
    ---------------
    Parameters:
    np_array: array to be detrended ordered by date with no missing values
    window (int): deafult 360 for 360day calendar, the trend to be subtracted 
    ---------------
    Returns: 
    (numpy array) detrended array

    '''
    diff = list()
    for i in range(360, len(np_array)):
        value = np_array[i] - np_array[i - 360]
        diff.append(value)
    return diff


def inverse_difference(last_ob, value):
    '''
    Apply on a vlue to retrend it after detrended and corrected
    ---------------
    Parameters:
    last_ob (float): observation that was subtracted from value in detrending
    value (float) : current value after detrending and any other corrections (e.g bias correction)
    ---------------
    Returns:
    (float) current value retrended 

    '''
    return value + last_ob


def mean_bias_correct(model_data, obs, ref_times, future_times):
    '''
    Mean bias correction
    ---------------
    Parameters: 
    model (xarray df): climate model xarray including time, lat, lon dimensions and xarray values representing the ones to be corrected
    obs (xarray df): including time, lat, lon dimensions 
    ref_times (tuple of strings): e.g. ('2000-01-01', '2010-01-01')
    future_times (tuple of strings): e.g. ('2020-01-01', '2030-01-01')
    ---------------
    Returns: 
    (xarray df) mean corrected future data
    '''

    # Select future data to be correcte
    try:

        # Select reference arrays
        past_model = model.sel(time=slice(*ref_times))
        past_obs = obs.sel(time=slice(*ref_times))
        future_model = model.sel(time=slice(
            *future_times)).reduce(np.mean, ('lat', 'lon'))

        # Bias
        past_model_mean = past_model.reduce(np.mean, ('lat', 'lon', 'time'))
        past_obs_mean = past_obs.reduce(np.mean, ('lat', 'lon', 'time'))
        bias = past_model_mean - past_obs_mean
        future_bias_corrected = future_model - bias
    except:
        print('unable to bias correct')
        return None

    return future_bias_corrected


def ecdf_bias_correction(model, obs, ref_times, future_times):
    '''
    An implementation of the empirical cumulative distribution function mapping 
    
    (predicted observed x) = F_m'(^-1)[F_m(X_o)]: future observations are equal to past observations projected on ecdf of past climate model, 
    then taking the inverse of future climate model ecdf.

    Main assumption: "the change from present-day to future in
    the observation distribution will be the same as the
    change in the model distribution" 
    -------------
    Parameters: 
    model (xarray df): climate model xarray including time, lat, lon dimensions and xarray values representing the ones to be corrected
    obs (xarray df): including time, lat, lon dimensions 
    ref_times (tuple of strings): e.g. ('2000-01-01', '2010-01-01')
    future_times (tuple of strings): e.g. ('2020-01-01', '2030-01-01')
    --------------
    Returns: 
    (xarray) with values ecdf corrected and retrended with time dimension equivalent to future_times minus the first 360 days


    '''

    try:

        model1 = match_calendar(model.sel(time=slice(*ref_times)), False)
        model_past = detrend_seasonality(
            (model1.reduce(np.mean, ('lat', 'lon'))).values)
        model_future = match_calendar(
            model.sel(time=slice(*future_times)), False)
        model_p_time = model_future.time[360:]
        model_future_obs = detrend_seasonality(
            (model_future.reduce(np.mean, ('lat', 'lon'))).values)
        sliced_obs = obs.sel(time=slice(*ref_times))
        observations = match_calendar(
            sliced_obs.reduce(np.mean, ('lat', 'lon')))
        detrend_obs = detrend_seasonality(observations.values)

        # Construct ECDF for historical model times
        vals_past, ecdf_model_past = construct_ecdf(model_past)
        vals_future, ecdf_model_future = construct_ecdf(model_future_obs)
        corrected = [inverse_ecdf_pr(fut, vals_future, np.array(ecdf_model_future)) for fut in [
            ecdf_val(obs, vals_past, ecdf_model_past) for obs in detrend_obs]]
        corrected_retrended = [inverse_difference(
            observations.values[i], corrected[i]) for i in range(len(corrected))]
        data = xr.DataArray(corrected_retrended,
                            dims=('time'),
                            coords={'time': model_p_time})

    except Exception as e:
        print(e)
        print('unable to bias correct')
        return None

    return(data)
