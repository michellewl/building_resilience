import numpy as np
import xarray as xr
from glob import glob
from itertools import chain

module load python3
module use /g/data3/hh5/public/modules
module load conda/analysis3

# Copy and paste to raijin

if __name__ == "__main__":
    # Retrieve ERA5 temperature data for 2018 
    year = 2018
    all_data = glob('/g/data3/ub4/era5/netcdf/surface/2T/{}/*.nc'.format(year))
    current_dataset = xr.open_mfdataset(all_data, chunks = {'time':10, 'latitude':100, 'longitude':100}, combine = 'by_coords')
    cropped_arr = current_dataset.sel(latitude = slice(72.0, 35.0), longitude = slice(-25.0, 40.0))
    tmax_array = cropped_arr.resample(time='1D').mean()
    tmax_array.to_netcdf(path = '/g/data3/r15/av3286/cyclone_shear/T2_{}_MEAN.nc'.format(year))
    print("Finished {}".format(year)) 
