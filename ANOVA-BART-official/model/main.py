from .btree import BTREE, BPTREE
import numpy as np
import pandas as pd
import yaml
import os
import copy
import csv

from .utils import *

import argparse
parser = argparse.ArgumentParser(description = '')
parser.add_argument('-n', '--experiment_name', type = str, default = '', help = 'experiment name')
parser.add_argument('--alpha', type = float, default = 0.95, help = 'splitting probability alpha')
parser.add_argument('--beta', type = float, default = 2., help = 'splitting probability beta')
parser.add_argument('--var_height', type = float, default = 1, help = 'variance for height prior')
parser.add_argument('--b_hyperparam1', type = float, default = -1, help = 'hyperparameter1 for b prior')
parser.add_argument('--b_hyperparam2', type = float, default = 1, help = 'hyperparameter2 for b prior')
parser.add_argument('--gamma_shape', type = float, default = 1, help = 'hyperparameter1 for gamma prior')
parser.add_argument('--gamma_scale', type = float, default = 1, help = 'hyperparameter2 for gamma prior')
parser.add_argument('--grow_prob', type = float, default = 0.4, help = 'GROW probability')
parser.add_argument('--prune_prob', type = float, default = 0.4, help = 'PRUNE probability')
parser.add_argument('--change_prob', type = float, default = 0.2, help = 'CHANGE probability')
args = parser.parse_args([])

def make_tree_log(tree: BPTREE, max_depth: int):
    log_string = ''
    for c in range(max_depth):
        if c < len(tree.structure):
            tmp_str = tree.structure[c]
            tmp_string = ','.join([str(tmp_str.variable), str(np.round(tmp_str.b.item(), 3)), str(np.round(tmp_str.gamma.item(), 3))])
            tmp_string += ','
        else :
            tmp_string = ',,,'
        log_string += tmp_string
    return log_string

def sample_trees(trees: list[BPTREE], data: np.ndarray, y: np.ndarray, config: dict, z: np.ndarray, sigma2: float) -> list[BPTREE]:
    # y : (n,) np array, z : (T_max,) np array
    p = data.shape[1]
    tree_forward = np.zeros((data.shape[0],))

    for tree in trees :
        tmp_forward = tree.forward(data)            # (n,)
        tree_forward = np.column_stack((tree_forward, tmp_forward))
    
    tree_forward = tree_forward[:, 1:]              # each column :  tree (n, M)
    tree_forward = tree_forward * z.reshape((1, -1))
    tmp_residual = y - np.sum(tree_forward, axis = 1).squeeze()
    new_trees = []

    # LOGGING
    if config['log_all']:
        new_trees_log = ''
    else :
        new_trees_log = {'GROW':0, 'PRUNE':0, 'CHANGE':0, 'STAY':0, 'INIT':0}

    for i, tree in enumerate(trees):
        new_tree = copy.deepcopy(tree)

        changed = False

        if z[i]:
            what_to_do = np.random.choice(['grow', 'prune', 'change'], size = 1, p = [config['grow_prob'], config['prune_prob'], config['change_prob']]).item()
            tmp_residual += new_tree.forward(data)
            if what_to_do == 'grow':
                if len(new_tree.structure) < config['max_depth']:
                    changed = new_tree.grow(tmp_residual, data)
                    new_tree.sample_height(tmp_residual, data)
            elif what_to_do == 'prune':
                if len(new_tree.structure) > 0 :
                    changed = new_tree.prune(tmp_residual, data)
                    new_tree.sample_height(tmp_residual, data)
            else :
                if len(new_tree.structure) > 0 :
                    changed = new_tree.change(tmp_residual, data)
                    new_tree.sample_height(tmp_residual, data)

            tmp_residual -= new_tree.forward(data)
            # LOGGING
            if changed :
                changed = what_to_do.upper()
            else :
                changed = 'STAY'

        else :
            new_tree = BPTREE(p, config, error_variance = sigma2)
            new_tree.init_from_prior(data)
            changed = 'INIT'

        new_trees.append(new_tree)

        if config['log_all']:
            new_trees_log += make_tree_log(new_tree, config['max_depth'])
            new_trees_log += ','.join([str(np.round(new_tree.height, 3)), changed, str(z[i])])
            new_trees_log += ','
        else :
            new_trees_log[changed] += 1
   
    if config['log_all']:
        new_trees_log = new_trees_log[:-1]
    else :
        new_trees_log = {k:v/len(trees) for k, v in new_trees_log.items()}

    return new_trees, new_trees_log

def main(data: np.ndarray, y: np.ndarray, config: dict, test_data: np.ndarray, test_y: np.ndarray, y_mean,y_std) -> list[list[BPTREE]]:

    config["y_mean"] = y_mean
    config["y_std"] = y_std
    
    test_y = (test_y -  y_mean)/y_std
    y = (y - y_mean)/y_std
    
    assert y.shape[0] == data.shape[0]
    assert len(y.shape) == 1

    # tree initialization
    tree_ensemble = []          # list[BPTREE]
    p = data.shape[1]
    T = config['max_ensembles']
    # z = np.ones((T,))
    if config['init_z_from_prior']:
        z = np.random.binomial(1, config['p_z'], size = (T,))
    else :
        z = np.ones((T,))
    new_sigma2 = 1.
    for t in range(T):
        tree_t = BPTREE(p, config, error_variance = new_sigma2)
        if config['init_tree_from_prior']:
            tree_t.init_from_prior(data)
        tree_ensemble.append(tree_t)
    
    # logging initialization
    if config['log_all']:
        initialize_excel(config)
    else :
        tree_filename = config['experiment_name'] + '_log'
        tree_log_path = os.path.join(config['experiment_name'], f'{tree_filename}.csv')
        tree_log = {'GROW':0., 'PRUNE':0., 'CHANGE':0., 'STAY':0., 'INIT':0, 'train RMSE':0., 'test RMSE': 0., 'error_var':0., 'T':200}
        tree_fieldnames = list(tree_log.keys())
        with open(tree_log_path, mode = 'w', newline = '', encoding= 'utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = tree_fieldnames)
            writer.writeheader()

    # sampling
    samples = [tree_ensemble]       # list[list[BPTREE]]
    sample_logs = []
    for i in range(config['num_samples']):
        # sampling trees
        tree_ensemble, tree_ensemble_log = sample_trees(trees = tree_ensemble, data = data, y = y, config = config, z = z, sigma2 = new_sigma2)
        samples.append(tree_ensemble)
        
        # evaluate
        res = np.zeros((data.shape[0],))
        test_res = np.zeros((test_data.shape[0],))
        for i_t, t in enumerate(tree_ensemble):
            t_result = t.forward(data) * z[i_t]
            res += t_result
            t_result_test = t.forward(test_data) * z[i_t]
            test_res += t_result_test

        i_residual = y - res
        i_rmse = np.sqrt(np.mean(i_residual**2)).item() * y_std
        i_test_rmse = np.sqrt(np.mean((test_y - test_res)**2)).item() * y_std

        z_sum = np.sum(z).item(); z_sum = int(z_sum)
        print(f'{i}th sample train rmse : {np.round(i_rmse,4)}, test rmse : {np.round(i_test_rmse,4)}, # ensemble : {z_sum}')
        
        # error variance update
        inv_gamma_shape = (config['inv_gamma_nu'] + data.shape[0])/2
        inv_gamma_scale = (config['inv_gamma_nu'] * config['inv_gamma_lambda'] + np.sum(i_residual**2).item())/2
        new_sigma2 = np.random.gamma(shape = inv_gamma_shape, scale = 1/inv_gamma_scale, size = (1,)).item()
        new_sigma2 = 1/new_sigma2
        for t in tree_ensemble:
            t.var_error = new_sigma2

        # logging2
        if config['log_all']:
            tree_ensemble_log = ','.join([str(np.round(i_rmse, 3)), str(np.round(i_test_rmse, 3)), str(np.round(new_sigma2, 3)), str(z_sum), tree_ensemble_log])
            sample_logs.append(tree_ensemble_log)
        else :
            tree_ensemble_log['train RMSE'] = np.round(i_rmse, 3); tree_ensemble_log['test RMSE'] = np.round(i_test_rmse, 3)
            tree_ensemble_log['error_var'] = np.round(new_sigma2, 3); tree_ensemble_log['T'] = z_sum
         
            with open(tree_log_path, mode = 'a', newline = '', encoding = 'utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=tree_fieldnames)
                writer.writerow(tree_ensemble_log)
        
        # z update
        log_z_posterior = -np.sum(i_residual**2).item()/(2*new_sigma2) - config['c_star']*z_sum*np.log(data.shape[0])
        # log_z_posterior = -np.sum(i_residual**2).item()/(2*new_sigma2)

        z_k = np.random.choice(np.arange(T), size = 1)
        new_z = copy.deepcopy(z)
        new_z[z_k] = 1 - new_z[z_k]
        new_res = np.zeros((data.shape[0], ))
        for new_i_t, t in enumerate(tree_ensemble):
            new_t_result = t.forward(data) * new_z[new_i_t]
            new_res += new_t_result
        new_i_residual = y - new_res
        new_z_sum = np.sum(new_z).item(); new_z_sum = int(new_z_sum)
        new_log_z_posterior = -np.sum(new_i_residual**2).item()/(2*new_sigma2) - config['c_star']*new_z_sum*np.log(data.shape[0])
        # new_log_z_posterior = -np.sum(new_i_residual**2).item()/(2*new_sigma2)

        # acceptance_rate = np.exp(new_log_z_posterior - log_z_posterior) * (((1-config['p_z'])/config['p_z'])**(z_sum-new_z_sum))
        if new_z_sum > z_sum :
            assert new_z_sum == z_sum + 1
            comb_ratio = new_z_sum/(config['max_ensembles']-z_sum)
        else :
            assert new_z_sum == z_sum - 1
            comb_ratio = (config['max_ensembles']-new_z_sum)/z_sum
        acceptance_rate = np.exp(new_log_z_posterior - log_z_posterior) * comb_ratio
        #print(f'z {new_z_sum - z_sum} w.p. {acceptance_rate}')
        dice = np.random.uniform(0, 1, size = (1,)).item()
        if dice < acceptance_rate :
            z = new_z

    if config['log_all']:
        append_row_from_string(sample_logs, config)

    return samples

# def inference(new_data: np.ndarray,  config: dict, new_y:np.ndarray, model: list[list[BPTREE]]):
#     new_y -= np.mean(new_y)
#     result = np.zeros((new_data.shape[0],))
#     for trees in model:         # trees: list[BPTREE]
#         res = np.zeros((new_data.shape[0],))
#         for tree in trees:      # tree : BPTREE
#             res += tree.forward(new_data)
#         result = np.column_stack((result, res))
#     result = result[:, 1:]
#     result = np.mean(result, axis = 1)
#     inference_rmse = np.sqrt(np.mean((new_y-result)**2))
#     print(f'inference rmse : {inference_rmse}')
#     return inference_rmse
    
    
def predict(new_data: np.ndarray, config: dict, model: list[list[BPTREE]]):
    result = np.zeros((new_data.shape[0],))
    for trees in model:         # trees: list[BPTREE]
        res = np.zeros((new_data.shape[0],))
        for tree in trees:      # tree : BPTREE
            res += tree.forward(new_data)
        result = np.column_stack((result, res))
    result = result[:, 1:]
    result = np.mean(result, axis = 1)*config["y_std"] + config["y_mean"]
    
    return result


if __name__ == '__main__':
    # configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None and k in config:
            config[k] = v
     
    new_var_height = config['var_height']/np.sqrt(config['max_ensembles'])
    # config['var_height'] /= np.sqrt(config['num_ensembles'])
    config['var_height'] = new_var_height
    print(f'config var_height : {config["var_height"]}')

    if os.path.exists(config['experiment_name']):
        raise Exception('check the experiment name.')
    else :
        os.makedirs(config['experiment_name'])

    with open(os.path.join(config['experiment_name'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys = False)     

    # assert config['num_ensembles'] <= config['max_ensembles']

    # --------------------------------------------------------------------
    # synthetic data
    np.random.seed(42)
    data = np.random.uniform(0, 1, size=(1000,10))
    y = 10*np.sin(np.pi*data[:, 0]*data[:,1])+20*(data[:,2]-0.5)**2+10*data[:,3]+5*data[:,4]
    error = np.random.normal(0, 1, size=(1000,))
    y += error

    t_data = np.random.uniform(0, 1, size = (1000, 10))
    t_y = 10*np.sin(np.pi*t_data[:, 0]*t_data[:,1])+20*(t_data[:,2]-0.5)**2+10*t_data[:,3]+5*t_data[:,4]
    t_error = np.random.normal(0, 1, size=(1000,))
    t_y += t_error
    # --------------------------------------------------------------------

    # # --------------------------------------------------------------------
    # data = pd.read_csv(config['data_path'], index_col = 0)
    # x_colnames = copy.deepcopy(data.columns.tolist())
    # x_colnames.remove(config['y_variable_name'])
    # y = data.loc[:, config['y_variable_name']]
    # data = data.loc[:, x_colnames]

    # from sklearn import preprocessing
    # from sklearn.model_selection import train_test_split
    # data, t_data, y, t_y = train_test_split(data, y)        

    # quantile_transformer = preprocessing.QuantileTransformer()
    # data = quantile_transformer.fit_transform(data)
    # t_data = quantile_transformer.transform(t_data)
    # y = y.to_numpy(); t_y = t_y.to_numpy()
    # --------------------------------------------------------------------

    print(f'train data : {data.shape[0]}, test_data : {t_data.shape[0]}')
    samples = main(data, y, config, t_data, t_y)


    # bayesian averaging inference result
    if config['do_inference']:
        used_sample = samples[-200::10]
        inference_rmse = inference(t_data, t_y, used_sample)

        # filename = config['experiment_name'] + '_log'
        # xlsx_filename = os.path.join(config['experiment_name'], f'{filename}.xlsx')
        # wb = load_workbook(xlsx_filename)
        # ws = wb.active
        # next_row = ws.max_row + 1
        # ws.cell(row=next_row, column=1, value = 'inference rmse :')
        # ws.cell(row=next_row, column=2, value = str(np.round(inference_rmse, 3)))
        # wb.save(xlsx_filename)
        inference_text = f'inference RMSE : {inference_rmse}'
        inference_file_path = os.path.join(config['experiment_name'], 'inference_result.txt')
        with open(inference_file_path, mode = 'w', encoding='utf-8') as f:
            f.write(inference_text)