# -*- coding: UTF-8 -*-

from IPython.display import display
from html import HTML
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np



def nice_display(element, names):
    '''
    Prints a nice version of a series/list/array on the screen
    ----------
    Parameters:
    element: series/pandas df/array/list
    names: column names to present on screen for provided element
    ----------

    '''
    display(HTML(pd.DataFrame(element, columns=names).to_html()))


def sns_correlation_per_class(df, class_col, corr_cols):
    '''
    Plot a correlation seaborn plot per class in calss_col -- randomly chosen 4
    -----------
    df (Pandas DataFrame)
    corr_cols (list of strings): cols in df that you want to check correlations of


    '''
    for cls in np.random.choice(df[class_col].unique(), 4):
        sns.set(style="white")

        corr_vars = df.loc[df[class_col] == cls, corr_cols]
        corr = corr_vars.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.suptitle('Class # {}'.format(cls))
        plt.show()


def check_different_ids_same_class(df, id_col, class_col, xaxis, yaxis, time_unit='mon', group=None):
    '''
    Plot a line chart for four randomly chosen IDs from a certain class
    ----------
    Parameters:
    df (Pandas DataFrame)
    id_col (string): e.g. building id
    class_col (string): e.g. site # 7 
    xaxis (string): e.g. date
    yaxis (string)
    time_unit (string): e.g. mon -- make sure this string is a name of one of your columns
    group (string): should you want to group before plotting, you might want to use same string as xaxis   

    '''
    site = np.random.choice(df[class_col].unique(), 1)[0]
    df_temp = df.loc[df[class_col] == site, :]
    df_temp.loc[:, id_col] = pd.to_numeric(df_temp[id_col])
    buildings = np.random.choice(df_temp[id_col].unique(), 4, replace=False)
    df_temp.loc[:, time_unit] = pd.to_numeric(df_temp[time_unit])
    if (group):
        df_temp = df_temp.groupby([id_col, group]).mean().reset_index()[
            [id_col, xaxis, yaxis]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for b in range(len(buildings)):
        axes[b % 2][b // 2].plot(df_temp.loc[df_temp[id_col] == buildings[b], xaxis],
                                 df_temp.loc[df_temp[id_col] == buildings[b], yaxis], c=np.random.rand(3,))
        axes[b % 2][b //
                    2].set_title('Class # {}, ID # {}'.format(site, buildings[b]))
        plt.subplots_adjust(hspace=0.45)



def plot_sns_multi_boxplots_one_class(df, x, y, title):
    '''
    One plot with multiple boxplots corresponding to x-axis
    ----------
    Parameters:

    title (string)



    '''
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    sns.boxplot(x=x, y=y, data=df, showfliers=False).set_title(title);
    plt.show()



def plot_sns_usages_multi_boxplots_multi_class(df, x, y, class_col):
    '''
    Multiple plots (randomly chosen 4 per class), 
    each consists of multiple boxplots corresponding to usage (x-axis)
    ----------
    Parameters:



    '''
    for cls in np.random.choice(df[class_col].unique(), 4):
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        print('class is: ', cls)
        sns.boxplot(x=x, y=y, data=df[df[class_col] == cls], showfliers=False)
        axes.axis(ymin=0, ymax=0.05)
        plt.show()


def line_subplots_per_class(df, x, y, plots_per_col, num_cols, class_col, ymin, ymax):
    '''



    '''
    fig, axes = plt.subplots(plots_per_col, num_cols,
                             figsize=(14, 10), dpi=100)
    plot_num = 0
    for cls in np.random.choice(df[class_col].unique(), 4):

        axes[plot_num % plots_per_col][plot_num // plots_per_col].plot(df.loc[(
            df[class_col] == cls), x], df.loc[df[class_col] == cls, y], c=np.random.rand(3,))
        axes[plot_num % plots_per_col][plot_num //
                                       plots_per_col].set_title('Class # {}'.format(cls))
        axes[plot_num % plots_per_col][plot_num //
                                       plots_per_col].axis(ymin=ymin, ymax=ymax)
        plt.subplots_adjust(hspace=0.45)
        plot_num += 1


def heatmap_percentage_of_zeros_per_class_and_time(df, zero_col, cls, time_unit='mon'):
    '''
    function to present as a heatmap all percentage of zeros for a certain column (zero_col) 
    as a function of time unit and class (cls)  
    ----------------
    Parameters:
    df (pandas DataFrame)
    zero_col (string): the column to check how many zeros are there
    cls (string): class to split the y-axis
    time_unit (string): the column representing the time unit in the df (x-axis)

    ----------------
    Returns:
    seaborn heatmap
    '''
    zeros = df[df[zero_col] == 0]
    p_zeros = zeros.groupby([time_unit, cls]).count() / \
        df.groupby([time_unit, cls]).count()
    p_zeros.fillna(0)
    p_zeros.reset_index(inplace=True)
    newf = p_zeros[[time_unit, cls, zero_col]].sort_index().pivot(
        index=cls, columns=time_unit, values=zero_col)
    newf.index = pd.to_numeric(newf.index, errors='coerce')
    fig, axes = plt.subplots(1, 1, figsize=(14, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(newf.sort_index(), cmap=cmap)
    plt.xlabel(time_unit, fontsize=14)
    plt.ylabel(cls, fontsize=14)
    plt.suptitle('Percentage of {} equal 0 per {} in 2016'.format(
        zero_col, time_unit))
    plt.show()


def describe_df_by_classes(df, class1, class2, cols_to_desc):
    '''
    Use the pandas describe function over windows of the dataframe extracted based on class1 and class 2
    -----------
    Parameters: 
    df (pandas Dataframe)
    class1 (string)
    class2 (string)
    cols_to_desc (list of strings)
    ------------

    '''
    desc = df.loc[:, cols_to_desc]
    for cls1 in desc[class1].unique():
        for cls2 in desc[class2].unique():
            print('Site # {}, primary use {}'.format(cls1, cls2))
            print('-------')
            nice_display(HTML(desc.loc[(desc[class1] == cls1) & (
                desc[cls2] == cls2), cols_to_desc].describe()))
            print('-------')


def missing_per_col(df):
    '''
    
    
    '''
    miss = df.isnull().sum() / df.shape[0]
    nice_display((miss[miss > 0].sort_values(ascending = False)), names = ["Missing values percentage"])
