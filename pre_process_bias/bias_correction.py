# -*- coding: UTF-8 -*-
import xarray as xr
from scipy.signal import detrend
import numpy as np
import dataprocessing as dp
from statsmodels.distributions.empirical_distribution import ECDF
import iris 

def remove_leaps_xarray(xr):
    '''
    Remove all 29-02 days
    '''
    return xr.where(((xr.time.dt.day != 29) | (xr.time.dt.month != 2)), drop=True) 
        

def convert_units(xr):
    cube = xr.to_iris()
    return cube.convert_units('celsius')



def detrend_seasonality(np_array):
    '''

    '''
    diff = list()
    for i in range(len(np_array) - 1, 364, -1):
        value = np_array[i] - np_array[i - 365]
        diff.append(value)
    return diff

def inverse_difference(last_ob, value):
    return value + last_ob



def mean_bias_correct(model_data, observations, ref_times, future_times):
    '''
    Mean bias correction
    '''



    # Select future data to be correcte
    try:

        # Select reference arrays
        past_model = model_data.sel(time=slice(*ref_times))
        past_obs = observations.sel(time=slice(*ref_times))
        future_model = model_data.sel(time=slice(
            *future_times)).reduce(np.mean, ('lat', 'lon'))

        # Bias
        past_model_mean = past_model.reduce(np.mean, ('lat', 'lon', 'time'))
        past_obs_mean = past_obs.reduce(np.mean, ('lat', 'lon', 'time'))
        bias = past_model_mean - past_obs_mean
        future_bias_corrected = future_model - bias
    except:
        print('unable to bias correct')
        return None

    return(future_bias_corrected)


def delta_change_correct(model_data, observations, ref_times, future_times):
    '''
    Delta change correction
    '''

    # Select reference arrays
    past_model = model_data.sel(time=slice(*ref_times))
    past_obs = observations.sel(time=slice(*ref_times))

    # Select future data to be corrected
    future_model = model_data.sel(time=slice(*future_times))

    # Bias
    past_model_mean = past_model.mean(dim='time')
    future_model_mean = future_model.mean(dim='time')
    diff = future_model_mean - past_model_mean

    obs_bias_corrected = past_obs + diff

    return(obs_bias_corrected)


def ecdf_bias_correction(model, obs, ref_times, future_times):
    
    # Subset past and future
    
    model_h = detrend_seasonality(remove_leaps_xarray(model.sel(time=slice(*ref_times)).reduce(np.mean, ('lat', 'lon'))).values)
    
    model_f = model.sel(time=slice(*future_times))

    model_p_time = remove_leaps_xarray(model_f).time[365:] 
    
    last_ob_m = remove_leaps_xarray((model_f).reduce(np.mean, ('lat', 'lon'))).values

    model_p = detrend_seasonality(last_ob_m)

    obs_h = detrend_seasonality(remove_leaps_xarray(obs.sel(time=slice(*ref_times)).reduce(np.mean, ('lat', 'lon'))).values)

    # Construct ECDF for historical model times
    ecdf_m_h = ECDF(model_h)

    # Find x_{m, h} mean
    mu_m_h = np.mean(model_h)

    # Find x_{m, p} mean
    mu_m_p = np.mean(model_p)

    r1 = mu_m_h / mu_m_p
    r2 = mu_m_p / mu_m_h

    corrected = [np.quantile(obs_h, (ecdf_m_h(r1*val)))*r2
                 for val in model_p]

    corrected_retrended  = [inverse_difference(last_ob_m[i], corrected[i]) for i in range(len(corrected))]


    # concat_xr = xr.concat((corrected), dim = "height") 
    data = xr.DataArray(corrected_retrended,
                    dims=('time'),
                     coords={'time': model_p_time})
    return(data)
