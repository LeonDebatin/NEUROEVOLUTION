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
from scipy.stats import wilcoxon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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



def wilcoxon_test(series1, series2):
    return wilcoxon(series2, series1, alternative='less')[1]

def get_wilcoxon_p_values(data):

    p_values = pd.DataFrame(np.zeros((len(data), len(data))), index=data.keys(), columns=data.keys())

    for key1 in data.keys():
        for key2 in data.keys():
            if key1 != key2:
                p_values.loc[key1, key2] = wilcoxon_test(data[key1], data[key2])
    
    return p_values

def color_wilcoxon(val):
    if val == 0:
        color = 'black'
    elif val > 0.05:
        color = 'lightcoral'
    else:
        color = 'mediumseagreen'
    return f'background-color: {color}'



def generate_niter_plots(dataframes, titles, target):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle("Median Train/Val RMSE for " + f'{target}' , fontsize=12)

    for df, ax, title in zip(dataframes, axes.flatten(), titles):
        grouped_data = df.groupby('iterations').median().reset_index()
        
        sns.lineplot(x=grouped_data['iterations'], y=grouped_data['train_score'], label='Training Score', linestyle='-', color='red', ax=ax)
        sns.lineplot(x=grouped_data['iterations'], y=grouped_data['val_score'], label='Validation Score', linestyle='--', color='blue', ax=ax)

        ax.set_title(title)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Score')
        ax.set_ylim(0,3)
        ax.legend()

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    

def filter_dataframe(df, gp_best_params=None, gs_gp_best_params=None, ne_best_params=None, neat_best_params=None):
    
    if gp_best_params is not None:
        n_iter = gp_best_params['n_iter']
    elif gs_gp_best_params is not None:
        n_iter = gs_gp_best_params['n_iter']
    elif ne_best_params is not None:
        n_iter = ne_best_params['n_iter']
    elif neat_best_params is not None:
        n_iter = neat_best_params['n_iter']
    
    filtered_data = df[df['iterations'] <= n_iter]
    
    return filtered_data.head(n_iter * 10)