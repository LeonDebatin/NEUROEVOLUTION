
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION/gpolnel'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from gpolnel.problems.inductive_programming import SML
from gpolnel.algorithms.genetic_algorithm import GeneticAlgorithm
from NEUROEVOLUTION.utils import seed, clipp, cv_logger_init, cv_logger_append

import torch
import random
random.seed(seed)
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



def gsgp_train(dl_train, dl_val, ps, n_iter, initializer, sspace, selection_method, mutation_method, mutation_prob, xo_prob, xo_method, has_elitism, allow_reproduction, log_path, ffunction, seed, device):
    
    pi_sml = SML(
        sspace=sspace,
        ffunction=ffunction,
        dl_train=dl_train, dl_test=dl_val,
        n_jobs=8
    )
    mheuristic = GeneticAlgorithm(
        pi=pi_sml,
        initializer=initializer,
        selector= selection_method,
        crossover=xo_method,
        mutator= mutation_method,
        pop_size=ps,
        p_m=mutation_prob,
        p_c=xo_prob,
        elitism=has_elitism,
        reproduction=allow_reproduction,
        device=device,
        seed=seed
    )
    mheuristic._initialize()
    mheuristic.solve(
        n_iter,
        verbose=0, log=1, log_path=log_path,
        test_elite=True
    )
    return mheuristic


def gsgp_cross_validation(X, y, batch_size, shuffle, kfold, initializer, ps, n_iter,  sspace,  selection_method, mutation_prob, mutation_method, xo_prob, xo_method, has_elitism, allow_reproduction,log_path_cv, log_path_train, ffunction, seed,  device, id=None):
    results = []
    
    cv_logger_init(log_path_cv)
            
    for fold, (train_index, val_index) in enumerate(kfold.split(X)):
        X_train, X_val = X.loc[train_index], X.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        
        X_train, X_val = clipp(X_train, X_val)
        
        ds_train = TensorDataset(torch.tensor(X_train.values), torch.tensor(y_train.values))
        ds_val = TensorDataset(torch.tensor(X_val.values), torch.tensor(y_val.values))

        dl_train = DataLoader(ds_train, batch_size, shuffle)
        dl_val = DataLoader(ds_val, batch_size, shuffle)

        m = gsgp_train(dl_train = dl_train, dl_val = dl_val, initializer=initializer, ps=ps, n_iter=n_iter, sspace = sspace, selection_method=selection_method, mutation_method=mutation_method, xo_method=xo_method, ffunction=ffunction, log_path= log_path_train,  mutation_prob=mutation_prob, xo_prob=xo_prob, has_elitism=has_elitism, allow_reproduction=allow_reproduction,  seed=seed,  device=device)
        score = m.best_sol.fit
        results.append(score)
    
        cv_logger_append(log_path_cv, id, fold, score, m.best_sol.repr_)
        
    avg_score = np.mean(results)
    print('cv_score:', avg_score)
    
    return avg_score

