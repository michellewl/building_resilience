import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt 

def summarise_cbecs():
    '''
    Output a summary of the CBECS data
    Returns:
    --------
    counts_df: pd.Dataframe
        dataframe of statistics
    '''
    
    # read in CBECS data
    cbecs_data = pd.read_csv("cbecs_2003_2012.csv")

    uses = {0: 'All', 1:'Vacant',2:'Office',4:'Laboratory',5:'Nonrefrigerated warehouse',6:'Food sales',7:'Public order and safety',
        8:'Outpatient health care',11:'Refrigerated warehouse',12:'Religious worship',13:'Public assembly',14:'Education',
        15:'Food service',16:'Inpatient health care',17:'Nursing',18:'Lodging',23:'Strip shopping mall',24:'Enclosed mall',
        25:'Retail other than mall',26:'Service',91:'Other'}

    stats = ['buildings', 'heating', 'cooling',
            'elec_heat', 'ng_heat', 'fo_heat', 'dh_heat',
            'elec_cool', 'ng_cool', 'fo_cool', 'dh_cool']

    counts_df = pd.DataFrame(columns=stats, index=uses.values())

    # Subset buildings by primary heating/cooling fuel for each house type
    for use_key in uses.keys():

        use = uses[use_key]

        if use_key == 0:
            use_data = cbecs_data
        else:
            use_data = cbecs_data[cbecs_data['PBA']==use_key]

        counts_df.loc[use]['buildings'] = len(use_data)
        counts_df.loc[use]['heating'] = len(use_data[use_data['HT1']==1])
        counts_df.loc[use]['cooling'] = len(use_data[use_data['COOL']==1])

        counts_df.loc[use]['elec_heat'] = len(use_data[use_data['ELHT1']==1])
        counts_df.loc[use]['ng_heat'] = len(use_data[use_data['NGHT1']==1])
        counts_df.loc[use]['fo_heat'] = len(use_data[use_data['FKHT1']==1])
        counts_df.loc[use]['dh_heat'] = len(use_data[use_data['DHHT1']==1])

        counts_df.loc[use]['elec_cool'] = len(use_data[use_data['ELCOOL']==1])
        counts_df.loc[use]['ng_cool'] = len(use_data[use_data['NGCOOL']==1])
        counts_df.loc[use]['fo_cool'] = len(use_data[use_data['FKCOOL']==1])
        counts_df.loc[use]['dh_cool'] = len(use_data[use_data['DHCOOL']==1])  

    counts_df.to_csv("cbecs_counts.csv") 

    return counts_df

def summarise_rbecs():
    '''
    Output a summary of the RBECS data
    Returns:
    --------
    counts_df: pd.Dataframe
        dataframe of statistics
    '''

    # Read in RECS data
    rbecs_data = pd.read_csv("rbecs_2005_2009_2015.csv")

    uses = {1:'Mobile home', 2:'Detetched house', 3:'Attached house',
                 4:'Apartment (2-4 units)', 5:'Apartment (5+ units)'}

    stats = ['buildings', 'heating', 'cooling',
            'elec_heat', 'ng_heat', 'fo_heat']

    counts_df = pd.DataFrame(columns=stats, index=uses.values())

    # Subset buildings by primary heating/cooling fuel for each house type
    for use_key in uses.keys():

        use = uses[use_key]

        if use_key == 0:
            use_data = rbecs_data
        else:
            use_data = rbecs_data[rbecs_data['TYPEHUQ']==use_key]

        counts_df.loc[use]['buildings'] = len(use_data)
        counts_df.loc[use]['heating'] = len(use_data[use_data['HEATHOME']==1])
        counts_df.loc[use]['cooling'] = len(use_data[use_data['ELCOOL']==1])

        counts_df.loc[use]['elec_heat'] = len(use_data[use_data['FUEL_HEAT']==5])
        counts_df.loc[use]['ng_heat'] = len(use_data[use_data['FUEL_HEAT']==1])
        counts_df.loc[use]['fo_heat'] = len(use_data[use_data['FUEL_HEAT']==3])

    counts_df.to_csv("rbecs_counts.csv") 

    return counts_df





if __name__ == "__main__":

    print("Running...")
    
    cbecs_stats = summarise_cbecs()
    recs_stats = summarise_rbecs()

