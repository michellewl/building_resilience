# -*- coding: UTF-8 -*-
import dataprocessing as dp
import baspy as bp
import numpy as np
import pandas as pd
import datetime


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


def slice_lat_lon(xr_df, lon, lat):
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
    sliced_xr_temp = xr_df.sel(lon=slice(lon - 2, lon),
                               lat=slice(lat - 2, lat)).to_dataframe().reset_index()

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


def get_threshold_world(lati_st, lati_end, lon_st, lon_end, era=True, era_var='t2max', c_var='tas', catl_model=None, run=0, exper="", model=""):
    '''
    Get year/month/day from netcdf time column
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


    Returns:
    --------
    df (pd.DataFrame):
        with # days above different threshold for every lat/lon square (2X2) grouped by year/month/day
    '''

    longi = lon_st
    lati = 0
    curr_df = pd.DataFrame([])

    if(era):
        xr_temp = dp.load_era(era_var)
    else:
        xr_temp = catalog_to_xr(catl_model.T)
        xr_temp = xr_temp.sel(time=slice("1979-01", None))

    check_time()

    now = datetime.datetime.now()
    while (longi < lon_end):
        longi += 2
        if(lati == lati_end):
            check_time(now)
            now = datetime.datetime.now()
        lati = lati_st
        while (lati < lati_end):
            lati += 2
            sliced_xr_temp = slice_lat_lon(xr_temp, longi, lati)
            if (era):
                sliced_xr_temp = sliced_xr_temp.rename(columns={'year': 'yr'})
                grouped_temp = get_days_below_abv_thres_temp(sliced_xr_temp, 'MX2T')
            else:
                sliced_xr_temp = get_time_from_netcdftime(sliced_xr_temp)
                grouped_temp = get_days_below_abv_thres_temp(sliced_xr_temp, c_var)

            grouped_temp['run'] = run
            grouped_temp['model'] = model
            grouped_temp['lat_st'] = lati - 2
            grouped_temp['lat_end'] = lati
            grouped_temp['lon_st'] = longi - 2
            grouped_temp['lon_end'] = longi
            curr_df = curr_df.append(grouped_temp)

    curr_df.to_csv(str(model)+'_' + str(exper) + '_' +
                   str(run) + '_' + str(lon_st) + '.csv')
    return curr_df


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
        exper = cat_item.loc['Experiment']
        run = cat_item.loc['RunID']
        get_threshold_world(lati_st, lati_end, lon_st, lon_end, era=False, catl_model=pd.DataFrame(cat_item),
                            run=run, exper=exper, model=model)


# a bit of a bad code here - reading twice the catalog
def model_wrap(lati_st, lati_end, lon_st, lon_end):
    f_catalog = read_catalog()
    for model in f_catalog['Model'].unique():
        cube_wrap(lati_st, lati_end, lon_st, lon_end, model)
