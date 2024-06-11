#usage: py evaluation_script.py file_path_X_test file_path_y_test
#example: py evaluation_script.py "../NEUROEVOLUTION/datamart/data_project_nel.csv" "datamart/y_lactose.csv"

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from NEUROEVOLUTION.utils import root_mean_squared_error, drop_features, kfold, load_best_params

import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import sys
import csv
import numpy as np

from genetic_programming.utils import gp_crossover





def winsorize_iqr(fit_df, transform_df, multiplier=1.5):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
    Q1 = fit_df.quantile(0.25)
    Q3 = fit_df.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)

    # Apply clipping (Winsorizing) to each column
    df_clipped = transform_df.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return df_clipped


def eval_pred(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return mse,rmse, mae, mape, r2, corr


def main(file_path_X, file_path_y, target):
    X = pd.read_csv(file_path_X)
    X = drop_features(X)
    y = pd.read_csv(file_path_y)
    
    
    #total_batches = 1
    batch_size = X.shape[0]
    shuffle = True
    ffunction = Ffunctions('rmse')
    gp_best_params = load_best_params('genetic_programming/best_params/' + f'{target}' + '-best_params_final.pkl')
    
    gp_score = gp_crossover(X, y, batch_size, shuffle, kfold, gp_best_params['initializer'], gp_best_params['ps'], gp_best_params['n_iter'],  gp_best_params['sspace'],  gp_best_params['selection_method'], gp_best_params['mutation_prob, mutation_method, xo_prob, xo_method, has_elitism, allow_reproduction,log_path_cv, log_path_train, ffunction, seed,  device, id=None)















def mainxx(file_path_X_test, file_path_y_test):
    
    #Preprocessing
    print("Load and prepare Test set")
    
    X_test = pd.read_csv(file_path_X_test)
    y_test = pd.read_csv(file_path_y_test).to_numpy().ravel()
    
    X_train = pd.read_csv("datamart/X_train_untouched.csv")
    y_train = pd.read_csv("datamart/y_lactose_train.csv").to_numpy().ravel()
    mean = [np.mean(y_train) for _ in range(len(y_test))]
    
    columns_to_clip = ['dim', 'dry_days', 'rumination_min_day', 'milk_kg_day', 'colostrum_separated_kg']
    X_train_clipped = X_train.copy()
    X_train_clipped[columns_to_clip] = winsorize_iqr(X_train[columns_to_clip], X_train[columns_to_clip], multiplier=2)
    
    pt = PowerTransformer(method='yeo-johnson', standardize=True)    
    pt.fit_transform(X_train_clipped)
    
    X_test_clipped = X_test.copy()
    X_test_clipped[columns_to_clip] = winsorize_iqr(X_train[columns_to_clip], X_test[columns_to_clip], multiplier=2)
    X_test_clipped_scaled = pd.DataFrame(pt.transform(X_test_clipped), columns=X_test.columns)
    
    columns_to_drop = ['dry_days', 'delivery_age_years']
    X_test_clipped_scaled.drop(columns=columns_to_drop, inplace=True)

    
    #Evaluation
    
    #Eval Mean
    y_pred = mean
    mean_mse ,mean_rmse, mean_mae, mean_mape, mean_r2, mean_corr = eval_pred(y_test, y_pred)
    
    #Eval EN
    print('Evaluate Elastic Net')
    with open('models/BasicMLModels/LinearRegression', 'rb') as file:
        model = pickle.load(file)    
        
    y_pred = model.predict(X_test_clipped_scaled)
    en_mse ,en_rmse, en_mae, en_mape, en_r2, en_corr = eval_pred(y_test, y_pred)

    #Eval 
    
    
    
    
    
    
    
    
    
    
    print('Save Evaluation Table and print results')
    with open('evaluation_table.csv', 'w', newline='\n') as csvfile:
        w = csv.writer(csvfile, delimiter=';')
        w.writerow(['model'] + ['mean_squared_error'] +['root_mean_squared_error'] + ['mean_absolute_error'] + ['mean_absolute_percentage_error']+ ['r2_score'] + ['correlation'] )
        w.writerow(['Mean'] + [mean_mse] + [mean_rmse] + [mean_mae] + [mean_mape]+ [mean_r2] + [mean_corr])
        w.writerow(['LinearRegression'] + [en_mse] + [en_rmse] + [en_mae] + [en_mape]+ [en_r2] + [en_corr])
    
    
    with open('evaluation_table.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row)
    


if __name__ == "__main__":
    file_path_X = sys.argv[1]
    file_path_y = sys.argv[2]
    target = sys.argv[3]
    main(file_path_X, file_path_y)