# -*- coding: UTF-8 -*-
import xarray
from scipy.signal import detrend
import numpy as np 
import dataprocessing as dp


####df.reduce(func = np.mean, dim = ('lon', 'lat')) 
def mean_bias_correct(model_data, observations, ref_times, future_times):
    '''
    Mean bias correction
    '''

    # Select reference arrays
    past_model = model_data.sel(time=slice(*ref_times))
    print('past_model:', past_model.values)
    past_obs = observations.sel(time=slice(*ref_times))
    print('past_obs:', past_obs.values)


    # Select future data to be corrected
    future_model = model_data.sel(time=slice(*future_times)).reduce(np.mean, ('lat', 'lon'))

    # Bias
    past_model_mean = past_model.reduce(np.mean, ('lat', 'lon', 'time'))
    # print('past_model_mean:', past_model_mean)
    past_obs_mean = past_obs.reduce(np.mean, ('lat', 'lon', 'time'))
    # print('past_obs_mean:', past_obs_mean)
    bias = past_model_mean - past_obs_mean
    # print('bias:', bias)

    future_bias_corrected = future_model - bias
    # print('future values:', future_bias_corrected)

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
    past_model_mean = past_model.mean(dim = 'time')
    future_model_mean = future_model.mean(dim = 'time')
    diff = future_model_mean - past_model_mean

    obs_bias_corrected = past_obs + diff

    return(obs_bias_corrected)


def ecdf_bias_correction(model, obs, ref_times, future_times):
    # Subset past and future
    model_h = model.sel(time=slice(*ref_times)).values
    model_p = model.sel(time=slice(*future_times)).values
    obs_h = obs.sel(time=slice(*ref_times)).values
    
    # Construct ECDF for historical model times
    ecdf_m_h = ECDF(model_h)

    # Find x_{m, h} mean
    mu_m_h = model_h.mean()

    # Find x_{m, p} mean
    mu_m_p = model_p.mean()
    
    r1 = mu_m_h/mu_m_p
    r2 = mu_m_p/mu_m_h

    corrected = [np.quantile(obs_h, (ecdf_m_h(r1*val)))*r2
                     for val in model_p]
    return(corrected)



