import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold

def grid_search_xgbr(predictors, targets, unique_depths, unique_n_estimators, model):
    '''
    Implement grid search
    Parameters:
    -----------
    predictors: np.array
        feature matrix
    targets: np.array
        target vector
    unique_depths: np.array
        model max depths
    unique_n_estimators: np.array
        model ensemble size
    model:
        Model to be fitted
    Return:
    -------
    gsCV: GridSearchCV
        fitted grid search object
    '''
    kfs = KFold(n_splits=5, shuffle = True, random_state = 0) #0 for first ones
    gsCV = GridSearchCV(estimator= model,
                param_grid={'max_depth': unique_depths, 'n_estimators': unique_n_estimators},
                cv = kfs,
                verbose = 100)

    gsCV.fit(predictors, targets)
    return gsCV

def plot_gs_results(gsCV, unique_depths, unique_n_estimators, file_path):
    '''
    Plot grid search results
    Parameters:
    -----------
    gsCV: GridSearchCV
        fitted grid search object
    unique_depths: np.array
        model max depths
    unique_n_estimators: np.array
        model ensemble size
    file_path: String
        path to save file  
    '''
    depths = gsCV.cv_results_['param_max_depth']
    scores = gsCV.cv_results_['mean_test_score']

    for depth in unique_depths:
        depth_constant_scores = scores[depths == depth]
        plt.plot(unique_n_estimators, depth_constant_scores, label = 'depth = '+str(depth))

    plt.legend(loc = "lower right")
    plt.xlabel('ensemble size')
    plt.ylabel('R-squared score')

    plt.savefig(file_path, dpi = 300)

def get_targets_predictors(cool_heat, fuel_type, res = False, lim = False):
    '''
    Read in the required datasets
    Parameters:
    -----------
    cool_heat: String
        either 'cooling' or 'heating'
    fuel_type: String
        either 'elec', 'ng', 'dh'
    res: Bool
        True for residential 
    lim: Bool
        True for OSM predictor data only
    Returns:
    --------
    predictors: pd.Dataframe
        model predictors
    targets: pd.Dataframe
        model targets
    sc: StandardScalar
        fitted scaler
    names: list
        feature names 
    '''

    if res and lim:
        predictor_file_path = 'res_'+fuel_type+'_'+cool_heat+'_lim_predictors.pkl'
        target_file_path = 'res_'+fuel_type+'_'+cool_heat+'_use_targets.pkl'
    elif res:
        predictor_file_path = 'res_'+fuel_type+'_'+cool_heat+'_all_predictors.pkl'
        target_file_path = 'res_'+fuel_type+'_'+cool_heat+'_use_targets.pkl'
    elif lim:
        predictor_file_path = fuel_type+'_'+cool_heat+'_lim_predictors.pkl'
        target_file_path = fuel_type+'_'+cool_heat+'_use_targets.pkl'
    else:
        predictor_file_path = fuel_type+'_'+cool_heat+'_all_predictors.pkl'
        target_file_path = fuel_type+'_'+cool_heat+'_use_targets.pkl'

    predictors = pd.read_pickle('../make_data_files/'+predictor_file_path)
    targets = pd.read_pickle('../make_data_files/'+target_file_path)

    # Filter out unnecessary predictors
    if res:
        predictors = predictors.reset_index(drop = True)
        targets = targets.reset_index(drop = True)
    else:
        predictors = predictors.reset_index(drop = True)
        targets = targets.reset_index(drop = True)
        predictors = predictors.drop(['ELCNS', 'ELEXP', 'MFEXP'], axis = 1)

    # Fix up the indexing
    ind_keys = [key for key in predictors.keys() if key.startswith('Unnamed')]
    predictors.drop(ind_keys, axis = 1, inplace = True)

    # Convert to KWH
    if (not res) or fuel_type == "ng":
        targets = targets/3.412

    # Normalise by area
    if res:
        if cool_heat == 'cooling':
            predictors.drop('TOTHSQFT', inplace = True, axis = 1)
            targets = targets/predictors['TOTCSQFT']
        else:
            predictors.drop('TOTCSQFT', inplace = True, axis = 1)
            targets = targets/predictors['TOTHSQFT']
    else:
        targets = targets/predictors['SQFT']

    # Scale the features
    sc = StandardScaler()
    names = predictors.keys()
    predictors = sc.fit_transform(predictors)

    return predictors, targets, sc, names