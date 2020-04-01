import pandas as pd
import numpy as np
from tqdm import tqdm
import os 


os.chdir('./resources')
import extra_resources as extra
os.chdir('../')

#---------------------------------------------------KEY FUNCTIONS------------------------------------------------------------------

def choose_UN_scenario(data_path, un_pop_data, scenario):
# Usage: un_pop_data is the data read form the UN population dataset
#        scenario is one of the following cases predicted by the UN:
#        'Constant fertility', 'Constant mortality', 'Instant replacement'
#        'High', 'Medium', 'Low', 'Momentum', 'No change', 'Zero migration'
#        'Lower 95 PI', 'Lower 80 PI', 'Upper 80 PI', 'Upper 95 PI', 'Median PI'

    # Check which type of scenario:
    if (scenario == 'Medium'):
        yrs_count = 151
        base = 1950
    elif (scenario == 'Lower 95 PI')or(scenario == 'Lower 80 PI')or(scenario == 'Upper 80 PI')or(scenario == 'Upper 95 PI')or(scenario == 'Median PI'):
        yrs_count = 17
        base = 2020
    else:
        yrs_count = 81
        base = 2020

    # Only use the data of interest for the purpose
    x = un_pop_data.loc[un_pop_data['Variant'] == scenario]
    x = x.drop(['LocID','VarID', 'Variant', 'MidPeriod', 'PopMale','PopFemale', 'PopDensity'], axis = 1)
    location_list = list(x['Location'].unique())
    location_df = pd.DataFrame(location_list, columns =['Country Name'])
    
    # Read teh country codes for the WB format and delete unwanted territories
    WBs = pd.read_csv(data_path + '/help_data/UN_country_code.csv')
    WBs = WBs.dropna(axis = 0).reset_index(drop=True)
    check_countries = location_df.merge(WBs, on='Country Name', how='left')
    check_countries = check_countries.dropna(axis = 0).reset_index(drop=True)

    # Insert empty columns in the DataFrame
    check_countries
    for i in range(yrs_count):
        empt_column = [0]*len(check_countries)
        if (yrs_count != 17):
            check_countries[str(i+base)] = empt_column
        else:
            check_countries[str(i*5+base)] = empt_column

    # Insert the actual values in the empty columns
    for j in range(len(check_countries)):
        country = check_countries['Country Name'].iloc[j]
        kappa = x.loc[x['Location'] == country]
        kappa = kappa['PopTotal']
        kappa = list(kappa)
        check_countries.iloc[j,2:] = kappa
        
    print(scenario + ' scenario population prediction data succesfully extracted in your variable!')
    return check_countries
#---------------------------------------------------------------------------------------------------------------------

def drop_unwanted_UN_years(UN_scenario_dataset):
    # drop unwanted column from the population dataset:
    # drop years prior to 2000 and after 2030
    drop_list = []
    for i in range(50):
        drop_list.append(str(i+1950))
    for i in range(70):
        drop_list.append(str(i+2031))
        
    UN_scenario_dataset = UN_scenario_dataset.drop(drop_list, axis = 1)
    return UN_scenario_dataset

#---------------------------------------------------------------------------------------------------------------------

def IEA_year(year):
    iea_year = year
    if (year < 2005) and (year >=2000): iea_year = 2000
    if (year < 2010) and (year >=2005): iea_year = 2005
    if (year < 2015) and (year >=2010): iea_year = 2010
    if (year < 2017) and (year >=2015): iea_year = 2015
    if (year >=2017): iea_year = 2017

    return iea_year

#---------------------------------------------------------------------------------------------------------------------

def get_year_of_WB_data(year,WB_data):
    
    ## The dataset from WorldBank has 216 countries (first 216 entries) and the rest of them are aggregated territories.
    countries_limit = 217 #
    year_string = str(year)+' [YR'+str(year)+']'

    #Set up new DF    
    countries_list = list(WB_data['Country Name'].unique()[0:countries_limit])
    country_code_list = list(WB_data['Country Code'].unique()[0:countries_limit])

    new_df = pd.DataFrame(countries_list, columns =['Country Name'])
    new_df['Country Code'] = country_code_list
    
    #Add the WB data into new DF
    series_list = list(WB_data['Series Name'].unique())
    for i in range(len(series_list)-1):
        series_name = series_list[i]
        WB_data_series = WB_data.loc[WB_data['Series Name'] == series_name].reset_index(drop=True)
        WB_data_series = WB_data_series.iloc[0:countries_limit,:]
        current_series_list = WB_data_series[year_string]
        new_df[series_name] = current_series_list
        
    print('WB data for the year '+str(year)+ ' was succesfully imported!')
    return new_df

#---------------------------------------------------------------------------------------------------------------------

def merge_WB_and_IEA(WB_data, IEA_data):
    merged_df = WB_data.merge(IEA_data, on = 'Country Code', how = 'left')
    merged_df = merged_df.drop(['Country Name_y'], axis = 1)
    merged_df = merged_df.rename(columns={"Country Name_x": "Country Name"})
    
    return merged_df

#---------------------------------------------------------------------------------------------------------------------

def extract_key_years(df):
    return df[['Country Name','2020','2025','2050','2075']]

#---------------------------------------------------------------------------------------------------------------------

def add_UN_country_codes(UN_pop_df):
    code_names_UN =  pd.read_csv(path + 'names_UN_with_codes.csv')
    UN_pop_df = code_names_UN.merge(UN_pop_df, on='Country Name', how='left')
    return UN_pop_df

#---------------------------------------------------------------------------------------------------------------------

def merge_n_clear_NaN_df(UN_pop_df_with_country_code, WB_df):

    merged_df = UN_pop_df_with_country_code.merge(WB_df, on='Country Code', how='left')
    merged_df = merged_df.drop('Country Name_y', axis =1)
    merged_df = merged_df.rename(columns={"Country Name_x": "Country Name"})
    merged_df = merged_df.replace(to_replace='..', value=np.nan)
    values = {'elec_access_2017': 100}    #assume that were is no data for electricity access, it is equzl to 100%
    merged_df = merged_df.fillna(value=values)
    merged_df = merged_df.dropna(axis=0)
    merged_df = merged_df.reset_index(drop=True)

    return merged_df

#---------------------------------------------------------------------------------------------------------------------

def stick_household_data(UN_pop_df):
    household = pd.read_csv('final_household_data.csv')
    household = household.drop('Ref_date_of_measurement', axis =1)

    new = household.merge(UN_pop_df, on='Country Name', how='left')
    return new

#---------------------------------------------------------------------------------------------------------------------

def get_climate_year(climate_data, year):
    year_of_interest = climate_data[climate_data['yr'] == str(year)]
    
    mid_lat_p = []
    mid_lon_p = []
    country_name = []
    
    for i in tqdm(range(len(year_of_interest))):
        mid_lat_p.append(np.mean((float(year_of_interest['lati_st'].iloc[i]),float(year_of_interest['lati'].iloc[i]))))
        mid_lon_p.append(np.mean((float(year_of_interest['longi_st'].iloc[i]),float(year_of_interest['longi'].iloc[i]))))
        country_name.append(extra.whichCountry((mid_lat_p[i],mid_lon_p[i])))
        
    year_of_interest['country name'] = country_name
    return year_of_interest

#---------------------------------------------------------------------------------------------------------------------

def get_country_CDD(country_code, dataset):
    
    points_in_country = list(dataset[dataset['country name'] == country_code]['CDD'])
    cumulative_cdd = 0
    for i in range(len(points_in_country)):
        cumulative_cdd += float(points_in_country[i])
        
    cumulative_cdd = cumulative_cdd/len(points_in_country)
    
    return cumulative_cdd