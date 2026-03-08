import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer

def load_fold(dataset_name: str, fold: int):
    xs = []
    ys = []
    for i in range(1, 6):
        xs.append(pd.read_csv(os.path.join('data', dataset_name, f'data_x{i}.csv')).to_numpy())
        ys.append(pd.read_csv(os.path.join('data', dataset_name, f'data_y{i}.csv')).to_numpy())

    fold_range = list(range(5)); fold_range.remove(fold)
    train_x = np.concatenate([xs[jx] for jx in fold_range])
    train_y = np.concatenate([ys[jy] for jy in fold_range])
    test_x = xs[fold]
    test_y = ys[fold]

    qt = QuantileTransformer(n_quantiles=min(train_x.shape[0], 1000), output_distribution='uniform', random_state=0)
    qt.fit(train_x)
    train_x = qt.transform(train_x)
    test_x = qt.transform(test_x)

    if dataset_name in ['abalone', 'boston_housing', 'boston', 'mpg', 'servo']:
        regression = True
    else :
        regression = False

    if regression:
        y_mean = np.mean(train_y).item()
        y_sd = np.std(train_y).item()
        train_y = (train_y - y_mean)/y_sd
        test_y = (test_y - y_mean)/y_sd
        return (train_x, train_y, test_x, test_y, y_mean, y_sd)
    else :
        return (train_x, train_y, test_x, test_y)
    