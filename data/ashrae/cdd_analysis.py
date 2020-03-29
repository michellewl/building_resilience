# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


def percentage_incharge_by_temp(df, cls_col, val_col, xaxis, temp=20):
    '''


    '''
    temp_to_meter = df.loc[df[xaxis] > temp, [cls_col, val_col, xaxis]]
    temp_to_meter = temp_to_meter.sort_values(
        by=xaxis, ascending=False, axis=0)
    temp_to_meter['cum_sum'] = temp_to_meter.groupby(
        cls_col)[val_col].transform(pd.Series.cumsum)
    temp_to_meter['percentage_incharge'] = 0
    for site in temp_to_meter['site_id'].unique():
        temp_to_meter.loc[temp_to_meter[cls_col] == site, 'percentage_incharge'] = (temp_to_meter.loc[temp_to_meter[cls_col] == site, 'cum_sum'] /
                                                                                    temp_to_meter.loc[temp_to_meter[cls_col] == site, val_col].sum())
    return temp_to_meter


def multivar_cum_dist_per_class(df, x, y, class_col):
    '''
    
    
    '''
    df_cdf = pd.DataFrame({class_col: [0], y: [0], x: [0], 'cdf':[0]})
    for site in df[class_col].unique():
        df1 = df[df[class_col] == site]
        n = df1.shape[0]
        for yi in sorted(df1[y].unique()):
            for xi in sorted(df1[x].unique()):
                cdf = df1[(df1[y] <= yi) & (df1[x] <= xi)].shape[0] / n 
                df_cdf = df_cdf.append(pd.DataFrame({'site':[site], y: [yi], x: [xi], 'cdf': [cdf]}))
    df_cdf = df_cdf.iloc[1:,]
    return df_cdf


def CDD_barplot(temp_to_meter, cls_col):
    for site in np.random.choice(temp_to_meter[cls_col].unique(), 4):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(45, 20)
        g = sns.barplot(temp_to_meter.loc[temp_to_meter['site_id'] == site, 'air_temperature'],
                        temp_to_meter.loc[temp_to_meter['site_id'] == site, 'percentage_incharge'])
        fig.suptitle('Temperature vs. Cumulative energy use at site # {}'.format(
            site), fontsize=14, fontweight='bold')

        ax.set_title("80% of energy above 20C˚ is consumed in temperatures {} and above".format(temp_to_meter.loc[(
            temp_to_meter['site_id'] == site) & (temp_to_meter['percentage_incharge'] >= 0.8), 'air_temperature'].iloc[0]))
        plt.axhline(0.8)
        plt.xticks(rotation=90)


