from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import csv
import pandas as pd
import torch
import pickle
import os
import sys
import numpy as np
import re

parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

parent_dir = os.path.abspath(os.path.join(os.path.dirname('../gpolnel'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NEUROEVOLUTION.gpolnel.operators.selectors import prm_tournament, prm_roulette_wheel, prm_rank_selection, prm_double_tournament
from NEUROEVOLUTION.gpolnel.operators.variators import swap_xo, prm_subtree_mtn, prm_hoist_mtn, prm_point_mtn, prm_gs_xo, prm_gs_mtn,nn_xo, prm_nn_mtn
from NEUROEVOLUTION.gpolnel.operators.initializers import ERC, grow, full, prm_grow

seed = 1

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device


def drop_features(X):
    columns_to_drop = ['dry_days', 'delivery_age_years']
    return X.drop(columns=columns_to_drop)


def clipp(X_train, X_test, multiplier=2):
    #we clip for 2 times interquantile range
    columns_to_clip = ['dim',  'rumination_min_day', 'milk_kg_day', 'colostrum_separated_kg'] #based on eda
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    X_train_clipped = X_train.copy()
    X_test_clipped = X_test.copy()
    X_train_clipped[columns_to_clip] = X_train_clipped[columns_to_clip].clip(lower=lower_bound, upper=upper_bound, axis=1)
    X_test_clipped[columns_to_clip] = X_test_clipped[columns_to_clip].clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    return X_train, X_test


def scale(X_train, X_test):
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    
    return X_train_scaled, X_test_scaled



def get_score(y, y_pred):
    return root_mean_squared_error(y, y_pred)


def cv_logger_init(path):
    with open(path, 'w', newline='\n') as csvfile:
                w = csv.writer(csvfile, delimiter=';')
                w.writerow(['id']+['fold']+ ['val_score']+ ['representation'])


def cv_logger_append(path, id, fold , val_score, representation):
    with open(path, 'a', newline='\n') as csvfile:
        w = csv.writer(csvfile, delimiter=';')
        w.writerow([id]+[fold]+[val_score]+[representation])
                


def save_best_params(path, best_params):
    with open(path, 'wb') as handle:
        pickle.dump(best_params, handle) #, protocol=pickle.HIGHEST_PROTOCOL

def load_best_params(path):
    with open(path, 'rb') as handle:
        best_params = pickle.load(handle)
    return best_params

def extract_number(tensor_str):
    match = re.match(r'tensor\(([\d.]+)', tensor_str)
    if match:
        return float(match.group(1))
    return None


def complete_params(best_params):
    if 'prm_tournament' in best_params['selection_method']:
        best_params['selection_method'] = prm_tournament(best_params['selection_pressure'])
        
    elif 'prm_roulette_wheel' in best_params['selection_method']:
        best_params['selection_method'] = prm_roulette_wheel()
    
    elif 'prm_rank_selection' in best_params['selection_method']:
        best_params['selection_method'] = prm_rank_selection()
        
    elif 'prm_double_tournament' in best_params['selection_method']:
        best_params['selection_method'] = prm_double_tournament(best_params['selection_pressure'], best_params['selection_pressure'])
        
    
    if 'prm_subtree_mtn' in best_params['mutation_method']:
        best_params['mutation_method'] = prm_subtree_mtn(initializer=prm_grow(sspace=best_params['sspace']))
        
    elif 'prm_gs_mtn' in best_params['mutation_method']:
        to, by = 5.0, 0.25
        ms = torch.arange(by, to + by, by, device=device)
        best_params['mutation_method'] = prm_gs_mtn(prm_grow(sspace=best_params['sspace']), ms)
    
    elif 'prm_nn_mtn' in best_params['mutation_method']:
        best_params['mutation_method'] = prm_nn_mtn(ms=best_params['mutation_step'], sspace=best_params['sspace'])
    
    elif 'prm_hoist_mtn' in best_params['mutation_method']:
        best_params['mutation_method'] = prm_hoist_mtn()
    
    elif 'prm_point_mtn' in best_params['mutation_method']:
        best_params['mutation_method'] = prm_point_mtn(sspace=best_params['sspace'], prob=best_params['mutation_prob'])
    
    
    if 'swap_xo' in best_params['xo_method']:
        best_params['xo_method'] = swap_xo
    
    elif 'prm_gs_xo' in best_params['xo_method']:
        best_params['xo_method'] = prm_gs_xo(prm_grow(sspace=best_params['sspace']), device=device)
    
    elif 'nn_xo' in best_params['xo_method']:
        best_params['xo_method'] = nn_xo
    
    return best_params
    
    
    
def mean_cross_validation(X, y, log_path_cv, id=None):
    results = []
    
    cv_logger_init(log_path_cv)
            
    for fold, (train_index, val_index) in enumerate(kfold.split(X)):
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        
        
        y_pred = [y_train.mean() for i in range(len(y_val))]
        score = get_score(y_val, y_pred)
        results.append(score)
    
        cv_logger_append(log_path_cv, id, fold, score, 'mean')
        
    avg_score = np.mean(results)
    print('cv_score:', avg_score)

    return avg_score