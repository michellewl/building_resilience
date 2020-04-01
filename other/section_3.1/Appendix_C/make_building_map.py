from grid_cell import *
from buildings import *
import ee
import xarray as xr
import pickle as pk
from global_land_mask import globe

if __name__ == "__main__":

    # Make a lat - lon grid
    lons = np.linspace(48, 149, 149)
    lats = np.linspace(10, 60, 221)

    # Make the lon/lat coordinates
    lon_array = np.zeros((lons.shape[0] - 1, lats.shape[0] - 1))
    lat_array = np.zeros((lons.shape[0] - 1, lats.shape[0] - 1))

    # Make a residential energy array
    res_energy_heat = np.zeros((lons.shape[0] - 1, lats.shape[0] - 1))
    res_energy_cool = np.zeros((lons.shape[0] - 1, lats.shape[0] - 1))
    
    # Make a commercial energy array
    comm_energy_heat = np.zeros((lons.shape[0] - 1, lats.shape[0] - 1))
    comm_energy_cool = np.zeros((lons.shape[0] - 1, lats.shape[0] - 1))

    # Load in the usage dataset
    print("Initialising land use dataset")
    ee.Initialize()
    land_cover = ee.ImageCollection("COPERNICUS/CORINE/V20/100m").filterDate('2017-1-1','2018-12-1')
    land_cover_cats = {}
    
    # Load in the temperature dataset
    print("Initialising heating and cooling degree days")
    hdd_cdd_data = xr.open_dataset('../hdd_cdd_data.nc')
    hdd_data = hdd_cdd_data['hdd']
    cdd_data = hdd_cdd_data['cdd']

    # Load the models and scalers
    com_cool_model = pk.load(open('../models/com_lim_elec_cool.pkl', 'rb'))
    com_cool_sc = pk.load(open('../models/com_lim_elec_cool_sc.pkl', 'rb'))
    com_heat_model = pk.load(open('../models/com_lim_ng_heat.pkl', 'rb'))
    com_heat_sc = pk.load(open('../models/com_lim_ng_heat_sc.pkl', 'rb'))
    res_cool_model = pk.load(open('../models/res_lim_elec_cool.pkl', 'rb'))
    res_cool_sc = pk.load(open('../models/res_lim_elec_cool_sc.pkl', 'rb'))
    res_heat_model = pk.load(open('../models/res_lim_ng_heat.pkl', 'rb'))
    res_heat_sc = pk.load(open('../models/res_lim_ng_heat_sc.pkl', 'rb'))

    # For each lat - lon grid cell, do:
    for i in range(lons.shape[0] - 1):

        lon_l = lons[i]
        lon_r = lons[i+1]

        for j in range(lats.shape[0] - 1):

            lat_b = lats[j]
            lat_t = lats[j+1]

            # get the lat and lon coordinates of the midpoint of the cell
            lon = lon_l + (lon_r - lon_l)/2
            lat = lat_b + (lat_t - lat_b)/2

            print("analysing cell lat = {}, lon = {}".format(lat, lon))

            lon_array[i, j] = lon
            lat_array[i, j] = lat

            if not globe.is_land(lat, lon):
                print("Water")
                res_energy_heat[i, j] = 0
                comm_energy_heat[i, j] = 0
                res_energy_cool[i, j] = 0
                comm_energy_cool[i, j] = 0
                continue

            # Get usage - if data not available, write zeros and continue
            p = ee.Geometry.Point([lon, lat])
            lc = land_cover.filterBounds(p).first()
            ee.Number(lc.reduceRegion(ee.Reducer.mode(),p,100)).getInfo()['landcover']
            usage = ee.Number(lc.reduceRegion(ee.Reducer.mode(),p,100)).getInfo()['landcover']

            # ignore if there is no data
            if not usage:
                print("No data")
                res_energy_heat[i, j] = 0
                comm_energy_heat[i, j] = 0
                res_energy_cool[i, j] = 0
                comm_energy_cool[i, j] = 0
                continue

            print("Land usage is {}".format(usage))

            # Get HDD and CDD
            hdd = hdd_data.sel(latitude = lat, longitude = lon, method = 'nearest')
            cdd = cdd_data.sel(latitude = lat, longitude = lon, method = 'nearest')

            print("Heating degree days = {}, Cooling degree days = {}".format(hdd, cdd))

            # Make a grid cell object
            grid_cell = GridCell(lat_b, lat_t, lon_l, lon_r, usage, hdd, cdd)

            # Get residential energy use
            if len(grid_cell.res_buildings) == 0:
                res_heat = 0
                res_cool = 0
            else:
                res_cool = grid_cell.get_res_energy(res_cool_model, res_cool_sc)
                res_heat = grid_cell.get_res_energy(res_heat_model, res_heat_sc)

            # Get commercial energy use
            if len(grid_cell.com_buildings) == 0:
                com_heat = 0
                com_cool = 0
            else:
                com_cool = grid_cell.get_com_energy(com_cool_model, com_cool_sc)
                com_heat = grid_cell.get_com_energy(com_heat_model, com_heat_sc)

            # Write values to arrays
            res_energy_heat[i, j] = res_heat
            comm_energy_heat[i, j] = com_heat
            res_energy_cool[i, j] = res_cool
            comm_energy_cool[i, j] = com_cool
