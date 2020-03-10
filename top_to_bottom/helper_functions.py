import pandas as pd
import numpy as np

#---------------------------------------------------KEY FUNCTIONS------------------------------------------------------------------

def choose_UN_scenario(un_pop_data, scenario):
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
    WBs = pd.read_csv('data/help_data/WB_country_code.csv')
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

    return check_countries

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