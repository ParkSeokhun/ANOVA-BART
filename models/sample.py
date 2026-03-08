from .model import BTREEs
from utils.tree_logging import *

import os
import numpy as np
import copy
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import invgamma
import statsmodels.api as sm
import yaml
import pickle
from datetime import date
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


def sampling(data: np.ndarray, y: np.ndarray, config: dict, test_data: np.ndarray, test_y : np.ndarray, verbose = 100,scaler='default'):
    """
    data, y, test_data, test_y 
    """
    
    if scaler == 'uniform':
        qt = QuantileTransformer(n_quantiles=min(1000, data.shape[0]), output_distribution='uniform', random_state=0)
        data = qt.fit_transform(data)
        test_data = qt.transform(test_data)
    elif scaler == 'minmax':
        qt = MinMaxScaler()
        data = qt.fit_transform(data)
        test_data = qt.transform(test_data)
    elif scaler == 'default':
        data = data
        test_data = test_data   

    y = y.squeeze().astype(np.float32)
    test_y = test_y.squeeze().astype(np.float32)

    if config['y_dist'] == 'normal':
        y_std = np.std(y) 
        y_std = float(y_std)
        y_mean = np.mean(y); y_mean = float(y_mean)
        config['y_std'] = y_std; config['y_mean'] = y_mean
        y = (y - y_mean).flatten() / y_std
        test_y = (test_y - y_mean).flatten() / y_std

        ols_data = sm.add_constant(data)
        ols_model = sm.OLS(y, ols_data)
        results = ols_model.fit()
        ols_error_var = results.mse_resid           # OLS sigma^2

        inv_gamma_lambda = (2*ols_error_var)/(invgamma.ppf(config['nui']['q_lambda'], a = config['nui']['inv_gamma_nu'], scale = 1) * config['nui']['inv_gamma_nu'])
        inv_gamma_lambda = float(inv_gamma_lambda)
        config['nui']['inv_gamma_lambda'] = inv_gamma_lambda
        # print(f"Normal Regression error variance prior ~ IG(shape = {config['nui']['inv_gamma_nu']}, scale = {config['nui']['inv_gamma_lambda']})")

    #FIXME:0911 
    if config["data_driven_sparsity"]:
        if config["y_dist"] == "ber":
            gb_model = GradientBoostingClassifier(max_depth = 5)
            gb_model.fit(data,y)
            feature_weight = gb_model.feature_importances_
            config['feature_weight'] = feature_weight
        if config["y_dist"] == "normal":
            gb_model = GradientBoostingRegressor(max_depth = 5)
            gb_model.fit(data,y)         
            feature_weight = gb_model.feature_importances_
            config['feature_weight'] = feature_weight    
        print('Finished feature weight computation!')
    else:
        config['feature_weight'] = np.array([1.]*data.shape[1])
        print('You are using uniform feature weight.')

    config['max_depth'] = min([config['max_depth'], data.shape[1]])
    today = date.today()
    today = today.strftime('%m%d')
    if config['y_dist'] == 'normal':
        expn_list = [today, 'c*', config['c_star'], 'd', config['max_depth'], 'T', config['T_max'],'h-eps', config['step_size'],\
                    'L', config['leapfrog_L'], 'nu', config['nui']['inv_gamma_nu'], 'M', config['M'],]
    else :
        expn_list = [today, 'c*', config['c_star'], 'd', config['max_depth'], 'T', config['T_max'], 'h-eps', config['step_size'],\
                    'L', config['leapfrog_L'], 'b0-eps', config['const_step_size'], 'M', config['M']]
    config['experiment_name'] = os.path.join(config['log_dir'], config['y_dist'], config['dataset_name'], 'fold'+str(config['fold']), '_'.join([str(el) for el in expn_list]))
    if not config['data_driven_sparsity']:
        config['experiment_name'] += '_unif_weight'
    ver = 0
    if os.path.exists(config['experiment_name']):
        exist = True
        while exist:
            ver += 1
            new_name = config['experiment_name'] + f"_ver{ver}"
            exist = os.path.exists(new_name)
        config['experiment_name'] = new_name
    os.makedirs(config['experiment_name'])

    # config logging
    new_config = {k: v for k, v in config.items() if k not in ['feature_weight', 'fold']}
    with open(os.path.join(config['experiment_name'], 'config.yaml'), 'w') as f:
        yaml.dump(new_config, f, sort_keys = False)     
    with open(os.path.join(config['experiment_name'], 'feature_weight'), 'wb') as f:
        pickle.dump(config['feature_weight'], f)
    

    print(f"SAMPLING started {config['experiment_name']}")
    my_model = BTREEs(data, config)
    samples = [my_model]
    for ss in range(config['num_samples']):
        new_model = copy.deepcopy(samples[-1])
        new_model.btrees_update(data, y, config)
        new_model.z_update(data, y, config)
        new_model.nui_update(data, y, config)

        if (ss+1) % verbose == 0:
            new_model.evaluate(ss, data, y, test_data, test_y, config, print_res = True)
        else :
            new_model.evaluate(ss, data, y, test_data, test_y, config, print_res = False)

        samples.append(new_model)

    return samples
