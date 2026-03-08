from models.model import BTREEs
from models.btree import ATREE, BTREE
import csv
import os
from collections import Counter
import numpy as np

def str_to_csv(str_list : list[str], file_name: str, header_str: str):
    """  
    str_list : str의 리스트, 각 str은 하나의 row 기록이며 쉼표로 구분.
    file_name : 파일 경로
    header_str : column명이 쉼표로 구분되어 작성된 것.
    """
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 헤더 작성
        header = header_str.split(',')
        writer.writerow(header)

        # 데이터 작성
        for row_str in str_list:
            row = row_str.split(',')
            writer.writerow(row)

def tree_info(tree: BTREE):
    """
    height 값, z 값, (variable, b) max_depth 개
    """
    height = tree.height
    z = tree.z
    variables = []; bs = []
    for atree in tree.structure:
        variables.append(atree.variable)
        bs.append(atree.b)
    return (height, z, variables, bs)

def make_trees_log(trees: BTREEs, config: dict, log_max_depth: int=5) -> str:

    assert len(trees.model) == config['T_max']      
    gpc_log = trees.gpc_log                         # 'gppgs'
    height_log = trees.height_log                   # 'H_H__H'

    trees_log = ''
    
    if config['y_dist'] == 'normal':
        trees_log += str(trees.nui['sigma2'])[:7] + ','
    else :
        trees_log += str(trees.nui)[:7] + ','

    for t, tree in enumerate(trees.model):
        gpc_t = gpc_log[t]     # g/p/c/s str
        height_t, z_t, variables, bs = tree_info(tree)
        height_updated = (height_log[t] == 'H')
        tree_log_t = ','.join([gpc_t, str(height_updated), str(height_t)[:5], str(z_t)])
        tree_log_t += ','

        depth_t = 0
        for v, b in zip(variables, bs):
            tree_log_t += ','.join([str(v), str(b)[:5]]) + ','
            depth_t += 1
        if depth_t < log_max_depth:
            tree_log_t += ',,' * (log_max_depth - depth_t)
        trees_log += tree_log_t

    trees_log = trees_log[:-1]
    return trees_log

def trees_log_to_csv(samples: list[BTREEs], config: dict):
    
    log_str_list = []
    samples_max_depth = 0
    for trees in samples:
        for tree in trees.model:
            if samples_max_depth < len(tree.structure):
                samples_max_depth = len(tree.structure)

    for trees in samples:
        trees_log_str = make_trees_log(trees, config, samples_max_depth)
        log_str_list.append(trees_log_str)

    file_path = os.path.join(config['experiment_name'], 'trees_log.csv')

    # header
    tree_header = ','.join(['GPCS', 'H', 'height', 'z'])+','
    for d in range(samples_max_depth):
        tree_header += f'var_{d+1},b_{d+1},'
    tree_header *= config['T_max']

    if config['y_dist'] == 'normal':
        header = 'error_var,' + tree_header[:-1]
    else :
        header = 'beta_0' + tree_header[:-1]

    str_to_csv(log_str_list, file_path, header)
        
def update_log_to_csv(samples: list[BTREEs], config: dict):

    # trees log str list
    trees_log_str_list = []
    for trees in samples:
        trees_log_str = ''

        trees_gpcs = Counter(trees.gpc_log)
        trees_gpcs = [trees_gpcs[i] for i in ['g', 'p', 'c', 's']]  # list[int]
        trees_log_str += ','.join([str(s) for s in trees_gpcs]) + ','

        trees_bg = Counter(trees.b_log)
        trees_bg = [trees_bg['Y']]
        trees_log_str += ','.join([str(s) for s in trees_bg]) + ','

        trees_height_updates = Counter(trees.height_log)['H']       
        trees_z_l1 = np.sum(trees.zs).astype(int).item()            
        trees_z_updated = trees.z_updated                           
        trees_train_metric = trees.train_metric; trees_test_metric = trees.test_metric      # float
        hzm_logs = [trees_height_updates, trees_z_l1, trees_z_updated, trees_train_metric, trees_test_metric]
        trees_log_str += ','.join([str(ss) for ss in hzm_logs])

        if config['y_dist'] == 'normal':
            trees_log_str += ',' + str(trees.nui['sigma2'])[:7]
        
        trees_log_str_list.append(trees_log_str)

    # file path
    # expn = config['experiment_name']
    # expn_ = expn.replace('-', '_')
    file_path = os.path.join(config['experiment_name'], 'update_log.csv')

    # header
    header = '#G,#P,#C,#S,#b,#H,|z|,z upd,train,test'
    if config['y_dist'] == 'normal':
        header += ',var_error'

    str_to_csv(trees_log_str_list, file_path, header)
            

