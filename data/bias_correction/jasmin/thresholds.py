# -*- coding: UTF-8 -*-
import bias_correction as bc
import dataprocessing as dp
import baspy as bp
import numpy as np
import xarray as xr
import pandas as pd
import datetime



def read_catalog(model=None, data='cmip5', freq='day', var='tasmax'):
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
        filled with the selected variable (e.g. tasmax)
    '''
    for index, row in catl_model.iterrows():
        ds = bp.open_dataset(row)

    df_xr = ds.tasmax
    return df_xr


def get_days_below_abv_thres_temp(df, var='tasmax', gr='yr', thres=24):
    '''
    get CDD value per year 

    Parameters:
    ----------
    df (pd.DataFrame): a data frame which includes a grouping time column (e.g. yr, mon), a column for max temperature and lat and lon columns 
    var (string): deafult tasmax. MX2T for era data; column name in df representing maximum temperature per day
    gr (string): deafult yr (year). varible to group on.
    thres (float): the desired threshold for CDD

    Returns:
    --------
    grouped_df (pd.DataFrame):
        dataframe which includes the lat/lon combination and the associated CDD per year in df
    '''

    df['cel'] = df[var] - 273.15

    df['CDD'] = df['cel'] - thres

    df = df[df['CDD'] >= 0]
    print('are there CDDs?', df)

    grouped_df = df.groupby([gr]).sum().reset_index()

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
    '''
    A support function for get_threshold_world, acts as a wrapper 
    that executes CDD calculation regardless if it is era data 
    or corrected climate model data, then organizes the dataframe stored in dfs[key]
    to have details about the lat/lon the model name  and run number.
    ---------------------
    Parameters: 
    dfs (dict): 
    params (tuple): must include the variables run (int), model (string), lati (int), 
    longi (int), c_var (string): variable to be aggregated to CDD 
    --------------------- 
    Returns: 
    (dict) with keys representing either era or different bias correction methods (e.g. quantile mapping)

    '''
    run, model, lati, longi, c_var = list(params)

    for key in (dfs.keys()):
        if (key != 'era'):
            c_var = dfs[key].columns[list(
                pd.Series(dfs[key].columns).str.startswith('bc'))][0]
            print(c_var)
        dfs[key] = get_time_from_netcdftime(dfs[key])
        dfs[key] = get_days_below_abv_thres_temp(dfs[key], var=c_var)
        dfs[key]['run'] = run
        dfs[key]['model'] = model
        dfs[key]['lati_st'] = lati - 2
        dfs[key]['longi_st'] = longi - 2
        dfs[key]['lati'] = lati
        dfs[key]['longi'] = longi
        print(dfs)

    return dfs


def bias_cor_methods(sliced_xr_temp, sliced_xr_obs, params):
    '''
    A support function for get_threshold_world.
    Bias correction quantile mapping.
    -----------------
    Parameters: 
    sliced_xr_temp (xarray df):
    sliced_xr_obs (xarray df): 
    params (tuple of vars):
    -----------------
    Returns: 
    (dict) containting a key for quantile mapping ("qm") with a vlaue for that key being pandas dataframe 
    with column specifying the bias corrected values 
 

    '''
    bias_cor_dict = {}
    model, run, exper, lat_st, lon_st = list(params)

    sliced_xr_temp_bc_qm = bc.ecdf_bias_correction(
        sliced_xr_temp, sliced_xr_obs, ('2000-01-01', '2010-01-01'), ('2020-01-01', '2030-01-01'))
    try:
        sliced_xr_temp_bc_qm.name = 'bc_qm'
        sliced_xr_temp_bc_qm.to_netcdf("NC/" + str(model) + '_' + str(run) + '_' + str(
            exper) + '_' + str(lat_st) + '_' + str(lon_st) + '_' + 'qm_bc.nc')
        bias_cor_dict['qm'] = sliced_xr_temp_bc_qm.to_dataframe().reset_index()
    except:
        print('failed on lat-lon: ' + str(lat_st) + '_' + str(lon_st))
        bias_cor_dict['qm'] = pd.DataFrame()

    return bias_cor_dict


def get_threshold_world(lati_st, lati_end, lon_st, lon_end, era=True, era_var='t2max', c_var='tasmax', catl_model=None, run=0, exper="", model=""):
    '''
    Get csv and nc files for corrected bias and aggregated to CDD per year and lat/lon or csv with CDD for the reanalysis (ERA) data

    Parameters:
    ----------
    lati_st (int): the starting latitude for analysis
    lati_end (int): the end latitude for analysis
    lon_st (int): the starting longitude for analysis
    lon_end (int): the end longitude for analysis
    era (boolean): is it a reanalysis execution (deafult True)
    era_var (string): the name of the variable to load from era, deafult (t2max - maximum temperature)
    c_var (string): the name of the variable to load from climate model, deafult (tasmax - maximum temperature)
    catl_model (pandas DataFrame): 1 row of a df representing the specific run of the chosen model, deafult None for era execution 
    run (int): deafult (0) for era execution, otherwise what run number of the model it is
    exper (string): experiment name, deafult (empty string) for era.
    model (string) : name of the climate model, deafult (empty string) for era.


    Returns:
    --------
    df (pd.DataFrame):
        with # CDD days above threshold (24c˚) for every lat/lon square (2X2) grouped by year/month/day

    Also, saves csv files with CDD days per year and lat/lon combo 
    '''

    longi = lon_st
    lati = 0
    dict_df = {'era': pd.DataFrame([]), 'bc_mean': pd.DataFrame(
        []), 'delta': pd.DataFrame([]), 'qm': pd.DataFrame([])}

    if(era):
        xr_temp = dp.load_era(era_var)
        c_var = 'MX2T'
    else:
        xr_obs = dp.load_era(era_var)
        xr_temp = catalog_to_xr(catl_model.T)
        xr_temp = xr_temp.sel(time=slice("1979-01", None))
        try:
            if (np.max(xr_temp.time.dt.year) < 2030):
                print(model, ' doesnt have appropriate time range')
                return None
        except:
            print(model, ' doesnt have dt')
            return None

    check_time()

    now = datetime.datetime.now()
    while (longi < lon_end):
        longi += 2
        if(lati == lati_end + 1):
            check_time(now)
            now = datetime.datetime.now()
        lati = lati_st
        while (lati < lati_end):
            lati += 2
            sliced_xr_temp = slice_lat_lon(xr_temp, longi, lati)
            if (era):
                sliced_xr_temp = sliced_xr_temp.reduce(np.mean, ('lat', 'lon'))
                sliced_xr_temp = sliced_xr_temp.to_dataframe(
                ).reset_index().rename(columns={'year': 'yr'})
                dfs = {}
                dfs['era'] = sliced_xr_temp
            else:
                sliced_xr_obs = slice_lat_lon(xr_obs, longi, lati)
                dfs = bias_cor_methods(
                    sliced_xr_temp, sliced_xr_obs, (model, run, exper, lati, longi))

            df_list = grouped_df(dfs, (run, model, lati, longi, c_var))
            for key in df_list.keys():
                dict_df[key] = dict_df[key].append(df_list[key])
            print(dict_df)

    for item in dict_df.keys():
        if(dict_df[item].shape[0] > 0):
            dict_df[item].to_csv(str(item) + '_' + str(model) + '_' + str(exper) + '_' +
                                 str(run) + '_lat: ' + str(lati_st) + '_longitude_' + str(lon_st) + '_' + str(lon_end) + '.csv')
    return dict_df


def cube_wrap(lati_st, lati_end, lon_st, lon_end, model):
    '''
    Get csv files with threshold for all the runs of a model.
    A wraper function for get_threshold_world
    Parameters:
    ----------

    lati_st (int): the starting latitude for analysis
    lati_end (int): the end latitude for analysis
    lon_st (int): the starting longitude for analysis
    lon_end (int): the end longitude for analysis
    model (string): the name of the model to be bias corrected 
    ----------
    Returns: 
    None, just saves the appropriate csvs and NC files from the get_threshold_world function

    '''
    catalog = read_catalog(model)
    for item in range(1):
        cat_item = catalog.iloc[item, :]
        experi = cat_item.loc['Experiment']
        runi = cat_item.loc['RunID']
        get_threshold_world(lati_st, lati_end, lon_st, lon_end, era=False, era_var='t2max',
                            c_var='tasmax', catl_model=pd.DataFrame(cat_item), run=runi, exper=experi, model=model)


def model_wrap(lati_st, lati_end, lon_st, lon_end):
    '''
    Run bias correction for all models on the JASMIN platfrom.
    A wraper function for cube_wrap. 

    Parameters:
    ----------

    lati_st (int): the starting latitude for analysis
    lati_end (int): the end latitude for analysis
    lon_st (int): the starting longitude for analysis
    lon_end (int): the end longitude for analysis
    ----------
    Saves csvs for all models and nc files for all bias corrected data

    '''
    f_catalog = read_catalog()
    print(f_catalog['Model'].unique())
    for model in f_catalog['Model'].unique():
        cube_wrap(lati_st, lati_end, lon_st, lon_end, model)
