from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import csv
import pandas as pd
import torch
import pickle



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

