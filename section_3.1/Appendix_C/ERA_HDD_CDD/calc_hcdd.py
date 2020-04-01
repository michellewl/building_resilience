import numpy as np 
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

def to_farenheit(val):
    '''
    Convert value in degrees Kelvin to Farenheit
    '''
    return((val - 273.15)*(9/5)+32)

def get_cdd_hdd(dataset):
    '''
    Calculate HDD and CDD given temperature
    Parameters:
    -----------
    dataset: xr.DataArray
        temperature dataset T(lat, lon, time)
    Return:
    -------
    temp_far: xr.DataArray
        dataset with new entries for HDD and CDD
    '''

    # Convert temperatures to farenheit
    temp_far = dataset.apply(to_farenheit)

    # Calculate CDD
    temp_far['cdd'] = 65 - temp_far['t2m']
    temp_far['cdd'] = temp_far['cdd'].where(temp_far['cdd']>0, 0)
    temp_far['cdd'] = temp_far['cdd'].sum(dim = 'time')

    # Calculate HDD
    temp_far['hdd'] = temp_far['t2m'] - 65
    temp_far['hdd'] = temp_far['hdd'].where(temp_far['hdd']>0, 0)
    temp_far['hdd'] = temp_far['hdd'].sum(dim = 'time')

    return(temp_far)

if __name__ == "__main__":

    # Open temperature data
    data_2018 = xr.open_dataset('T2_2018_MEAN.nc')
    data_2018 = get_cdd_hdd(data_2018)

    # write out HDD/CDD data to a netCDF
    data_2018.to_netcdf('hdd_cdd_data.nc')