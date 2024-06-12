
import os
import sys
import csv
import tqdm
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from NEUROEVOLUTION.utils import seed, scale, clipp, cv_logger_init, cv_logger_append

parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION/gpolnel'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    

import random
random.seed(seed)
import torch
torch.manual_seed(seed)

import torch.nn as nn
from torch import optim
    
from NEUROEVOLUTION.utils import kfold,seed, cv_logger_init, cv_logger_append, drop_features
    


def nn_train(X_train, y_train, X_val, y_val, model, loss_fn, optimizer_name, lr, weight_decay, n_epochs, batch_size, log_path):
    

    with open(log_path, 'a', newline='\n') as csvfile:
        w = csv.writer(csvfile, delimiter=';')
        w.writerow(['id', 'epoch', 'train_score', 'val_score'])
    
    # Reset model weights
    model.apply(reset_weights)
    optimizer = get_optimizer(model.parameters(), optimizer_name, lr=lr, weight_decay=weight_decay)
    batch_start = torch.arange(0, len(X_train), batch_size)
    
    for epoch in range(n_epochs):
        model.train()
        
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # Take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                
                # Forward pass
                y_pred = model(X_batch)
                loss = torch.sqrt(loss_fn(y_pred, y_batch))
                loss.backward()
                
                # Backward pass
                optimizer.step()
                optimizer.zero_grad()
                
        model.eval()
        train_pred = model(X_train)
        train_score = torch.sqrt(loss_fn(train_pred, y_train))
        
        val_pred = model(X_val)
        val_score = torch.sqrt(loss_fn(val_pred, y_val))
        

        with open(log_path, 'a', newline='\n') as csvfile:
            w = csv.writer(csvfile, delimiter=';')
            w.writerow([epoch, train_score, val_score])
        
        # Save best model
    
    
    return model



def nn_cross_validation(X_train, y_train, model, cv_log_path, train_log_path, loss_fn, optimizer_name, lr, weight_decay, n_epochs, batch_size,  kf, id=None ):

    cv_logger_init(cv_log_path)
            
    results = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        
        
        X_train_cross, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cross, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        
        
        X_train_cross, X_val = clipp(X_train_cross, X_val)
        X_train_cross, X_val = scale(X_train_cross, X_val)
        
        X_train_cross = torch.tensor(X_train_cross.values, dtype=torch.float32)
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_train_cross = torch.tensor(y_train_cross.values, dtype=torch.float32).reshape(-1, 1)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
        
        
        model = nn_train(X_train_cross, y_train_cross, X_val, y_val,  model, loss_fn, optimizer_name, lr, weight_decay, n_epochs, batch_size,train_log_path)
        
        val_pred = model(X_val)
        score = torch.sqrt(loss_fn(val_pred, y_val))

        cv_logger_append(cv_log_path, id, fold, score, 'repr')

        results.append(score.detach().numpy())
        
    avg_score = np.mean(results)
    print(avg_score)
    return avg_score
        


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        
def nn_evaluate(model, X, y, loss_fn):
    y_pred = model(X)
    return loss_fn(y_pred, y).detach().numpy()

def create_random_model():
    n_neurons_1 = random.randint(1,10)
    n_neurons_2 = random.randint(1,10)
    r1 = random.uniform(0,1)
    r2 = random.uniform(0,1)
    model = nn.Sequential(nn.Linear( 12, n_neurons_1))
    
    if r1 < 0.25:
        model.add_module('relu1', nn.ReLU())
        
    elif 0.25 < r1< 0.5:
        model.add_module('sigmoid1', nn.Sigmoid())
    
    if 0.5 < r1 < 0.75:
        model.add_module('linear', nn.Linear(n_neurons_1, n_neurons_2))
        
        if r2 < 0.25:
            model.add_module('relu2', nn.ReLU())
        elif 0.25 < r2 < 0.5:
            model.add_module('sigmoid2', nn.Sigmoid())
        model.add_module('output', nn.Linear(n_neurons_2, 1))
        return model
    
    
    model.add_module('output', nn.Linear(n_neurons_1,1))
    return model

def get_optimizer(model_params, optimizer_name, lr, weight_decay=0.0):

    if optimizer_name == 'Adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        return optim.Adadelta(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'ASGD':
        return optim.ASGD(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer name')

def random_optimizer():
    r = random.uniform(0,1)
    if r < 0.2:
        return 'Adam'
    elif 0.2 < r < 0.4:
        return 'SGD'
    elif 0.4 < r < 0.6:
        return 'Adadelta'
    elif 0.6 < r < 0.8:
        return 'ASGD'
    return 'RMSprop'
