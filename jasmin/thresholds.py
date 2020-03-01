# -*- coding: UTF-8 -*-
import dataprocessing as dp
import baspy as bp
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import bias_correction as bc  


def read_catalog(model=None, data='cmip5', freq='day', var='tas'):
    '''
    Read in catalogue of climate models
    Parameters:
    ----------
    model (string): 
        None deafult
        HadGEM2-CC, BNU-ESM, CanESM2, bcc-csm1 
    freq (string):
        deafult day
        other options - month, year
    data (string):
        deafult cmip5
    var (string):
        deafult tas


    Returns:
    --------
    data (pandas.DataFrame):
        catalogue of all runs from the selected model
    '''
    if(model is not None):
        catl_model = bp.catalogue(dataset=data, Model=model,
                                  Frequency=freq, Var=var).reset_index(drop=True)
    else:
        catl_model = bp.catalogue(dataset=data,
                                  Frequency=freq, Var=var).reset_index(drop=True)
    return catl_model


def catalog_to_xr(catl_model):
    '''
    convert catalogue enetry of climate model to xarray
    Parameters:
    ----------
    catl_model (pandas.DataFrame 1 row)

    Returns:
    --------
    data (Xarray):
        with latitude, longitude, time and height dimensions 
        filled with the selected variable (e.g. tas)
    '''
    for index, row in catl_model.iterrows():
        ds = bp.open_dataset(row)

    df_xr = ds.tas
    return df_xr


def get_days_below_abv_thres_temp(df, var='tas', gr='yr'):
    '''
    get # days above/below different thresholds

    Parameters:
    ----------
    df (pd.DataFrame)
    var (string): deafult tas. MX2T for era data
    gr (string): deafult yr (year). varible to group on.

    Returns:
    --------
    data (pd.DataFrame):
        with threshold columns (e.g. column 32 will contain # days above 32)
    '''
    df['cel'] = df[var] - 273.15

    for temperature in range(-10, 19, 1):
        df[temperature] = df['cel'] < temperature

    for temperature in range(22, 40, 1):
        df[temperature] = df['cel'] > temperature

    grouped_df = df.reset_index().groupby([gr]).sum().reset_index()

    return grouped_df


def slice_lat_lon(xr_df, loni, lati):
    '''
    slice a lat/lon square 2X2
    Parameters:
    ----------
    xr_df (Xarray)
    lon (float): slice taken from lon -2 to lon
    lat (float): slice taken from lat -2 to lat

    Returns:
    --------
    data (Xarray):
        with latitude, longitude specified in slice
    '''
    sliced_xr_temp = xr_df.sel(lon=slice(loni - 2, loni),
                               lat=slice(lati - 2, lati))

    return sliced_xr_temp


def check_time(now=None):
    '''
    Get starting time, print time elapsed if now exist

    Parameters:
    ----------
    now: datetime.datetime object

    '''
    try:
        print(datetime.datetime.now() - now)
    except:
        print('started at: ' + str(datetime.datetime.now()))


def get_time_from_netcdftime(df, yr=True, mon=False, day=False):
    '''
    Get year/month/day from netcdf time column
    Parameters:
    ----------
    df: (pandas.DataFrame with time column (netcdf format))
    yr : deafult True. boolean, include year characterisitc  
    mon : deafult True. boolean, include month characterisitc  
    day : deafult True. boolean, include day characterisitc  

    Returns:
    --------
    df (pd.DataFrame):
        with yr/mon/day characteristics
    '''
    if(yr):
        df['yr'] = df['time'].apply(lambda x: x.year)
    if (mon):
        df['mon'] = df['time'].apply(lambda x: x.month)
    if (day):
        df['day'] = df['time'].apply(lambda x: x.day)
    return df

def grouped_df(dfs, params):
    run, model, lat_st, lati_end, lon_st, lon_end, c_var = list(params)
    print(c_var)

    for key in (dfs.keys()):
        if (key != 'era'):
            c_var = dfs[key].columns[list(pd.Series(dfs[key].columns).str.startswith('bc'))][0]
        dfs[key] =  get_time_from_netcdftime(dfs[key])
        print('here:', dfs[key])
        dfs[key] = get_days_below_abv_thres_temp(dfs[key], var = c_var)
        print('after:', dfs[key])
        dfs[key]['run'] = run
        dfs[key]['model'] = model
        dfs[key]['lat_st'] = lat_st
        dfs[key]['lat_end'] = lati_end
        dfs[key]['lon_st'] = lon_st
        dfs[key]['lon_end'] = lon_end


    return dfs


def bias_cor_methods(sliced_xr_temp, sliced_xr_obs, params):
    bias_cor_dict = {}
    model, run, exper, lat_st, lon_st = list(params)

    sliced_xr_temp_bc_mean = bc.mean_bias_correct(sliced_xr_temp, sliced_xr_obs, ('2000-01-01', '2010-01-01'), ('2020-01-01', '2030-01-01'))
    sliced_xr_temp_bc_mean.name = 'bc_mean'
    sliced_xr_temp_bc_mean.to_netcdf(str(model) + '_' + str(run) + '_' + str(exper) + '_'  + str(lat_st) + '_' + str(lon_st) + '_' + 'mean_bc.nc')
    bias_cor_dict['bc_mean'] = sliced_xr_temp_bc_mean.to_dataframe().reset_index()
    # sliced_xr_temp_bc_delta = bc.delta_change_correct(sliced_xr_temp, sliced_xr_obs, ('2000-01-01', '2010-01-01'), ('2020-01-01', '2030-01-01'))
    # print('values after bc:', sliced_xr_temp_bc_mean.values)
    # future = sliced_xr_temp.sel(time=slice('2020-01-01', '2030-01-01')).time
    # sliced_xr_temp_bc_delta = sliced_xr_temp_bc_delta.assign_coords(time=future)
    # sliced_xr_temp_bc_delta.to_netcdf(str(model) + '_' + str(run) + '_' + str(exper) + '_'  + str(lat_st) + '_' + str(lon_st) + '_' + 'delta_bc.nc')

    return bias_cor_dict



def get_threshold_world(lati_st, lati_end, lon_st, lon_end, era=True, era_var='t2max', c_var='tas', catl_model=None, run=0, exper="", model=""):
    '''
    Get 
    Parameters:
    ----------
    lati_st: (pandas.DataFrame with time column (netcdf format))
    lati_end : deafult True. boolean, include year characterisitc  
    lon_st : deafult True. boolean, include month characterisitc  
    lon_end : deafult True. boolean, include day characterisitc
    era:
    era_var:
    c_var:
    catl_model:
    run:
    exper:
    model:
    bias_cor: mean, delta, delta_var, quantile


    Returns:
    --------
    df (pd.DataFrame):
        with # days above different threshold for every lat/lon square (2X2) grouped by year/month/day
    '''

    longi = lon_st; lati = 0
    dict_df = {'era': pd.DataFrame([]), 'bc_mean': pd.DataFrame([]), 'delta': pd.DataFrame([])}

    if(era):
        xr_temp = dp.load_era(era_var)
        c_var = 'MX2T'
    else:
        xr_obs = dp.load_era(era_var)
        xr_temp = catalog_to_xr(catl_model.T)
        xr_temp = xr_temp.sel(time=slice("1979-01", None))
        if (np.max(xr_temp.time.dt.year) < 2030):
            print(model, ' doesnt have appropriate time range') 
            return 1

    check_time()

    now = datetime.datetime.now()
    while (longi < lon_end):
        longi += 2
        if(lati == lati_end  + 1):
            check_time(now)
            now = datetime.datetime.now()
        lati = lati_st
        while (lati < lati_end):
            lati += 2
            sliced_xr_temp = slice_lat_lon(xr_temp, longi, lati)
            if (era):
                sliced_xr_temp = sliced_xr_temp.to_dataframe().reset_index().rename(columns={'year': 'yr'})
                print(sliced_xr_temp)
                dfs = {};
                dfs['era'] = sliced_xr_temp
            else:
                sliced_xr_obs = slice_lat_lon(xr_obs, longi, lati)
                dfs = bias_cor_methods(sliced_xr_temp, sliced_xr_obs, (model, run, exper, lati_st, lon_st))

            df_list = grouped_df(dfs, (run, model, lati_st, lati_end, lon_st, lon_end, c_var))
            for key in df_list.keys():
                dict_df[key] = dict_df[key].append(df_list[key])
                print(df_list[key])

    for item in dict_df.keys():
        if(dict_df[item].shape[0] > 0):
            dict_df[item].to_csv(str(item)+ '_' + str(model)+ '_' + str(exper) + '_' +
                        str(run) + '_' + str(lon_st) + '.csv')
    return dict_df


def cube_wrap(lati_st, lati_end, lon_st, lon_end, model):
    '''
    Get csv files with threshold for all the runs of a model
    Parameters:
    ----------

    lati_st:
    lati_end:
    lon_st:
    lon_end:
    model:

    '''
    catalog = read_catalog(model)
    for item in range(len(catalog)):
        cat_item = catalog.iloc[item, :]
        experi = cat_item.loc['Experiment']
        runi = cat_item.loc['RunID']
        get_threshold_world(lati_st, lati_end, lon_st, lon_end, era=False, era_var='t2max', catl_model=pd.DataFrame(cat_item),
                            run=runi, exper=experi, model=model)


# a bit of a bad code here - reading twice the catalog
def model_wrap(lati_st, lati_end, lon_st, lon_end):
    f_catalog = read_catalog()
    for model in f_catalog['Model'].unique():
        cube_wrap(lati_st, lati_end, lon_st, lon_end, model)
