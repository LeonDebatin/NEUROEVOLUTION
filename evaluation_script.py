#usage: py evaluation_script.py file_path_X_test file_path_y_test
#example: py evaluation_script.py "datamart/X_test_untouched.csv" "datamart/y_lactose_test.csv"

import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import sys
import csv
import numpy as np

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


def main(file_path_X_test, file_path_y_test):
    
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
    file_path_X_test = sys.argv[1]
    file_path_y_test = sys.argv[2]
    main(file_path_X_test, file_path_y_test)