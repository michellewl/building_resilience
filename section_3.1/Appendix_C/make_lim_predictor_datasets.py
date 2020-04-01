import numpy as np 
import pandas as pd 
from glob import glob

def write_lim_pred_dataset(fuel_type, cool_heat, res = False):
    '''
    Make datasets with only features available in Open Street Map
    Parameters:
    -----------
    fuel_type: String
        ng = natural gas, elec = electricity
    cool_heat: String
        data for heating or cooling
    res: Bool
        Residential or commercial buildings
    '''

    # Read in the all predictor file
    if not res:
        csv_path = fuel_type+'_'+cool_heat+'_all_predictors.csv'
    else:
        csv_path = 'res_'+fuel_type+'_'+cool_heat+'_all_predictors.csv'
    data = pd.read_csv(csv_path)

    # Subset predictors and write to a csv
    if not res:
        data = data[['SQFT', 'PBA', 'HDD65', 'CDD65']]
        data.to_csv(fuel_type+'_'+cool_heat+'_lim_predictors.csv')
    else:
        if cool_heat == "heating":
            data = data[['STORIES', 'TYPEHUQ', 'TOTHSQFT', 'TOTCSQFT', 'HDD65', 'CDD65']]
        data.to_csv('res_'+fuel_type+'_'+cool_heat+'_lim_predictors.csv')

if __name__ == "__main__":

    # Write the residential data for different fuel types
    res_params = [('ng', 'heating'), ('elec', 'heating'), ('elec', 'cooling')]
    for p in res_params:
        write_lim_pred_dataset(*p, res = True)

    # Write the commercial data for different fuel types
    com_params = [('ng', 'heating'), ('elec', 'heating'), ('dh', 'heating'), ('elec', 'cooling')]
    for p in com_params:
        write_lim_pred_dataset(*p)