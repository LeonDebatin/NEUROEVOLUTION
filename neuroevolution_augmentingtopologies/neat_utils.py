
import sys
import os
import torch
import neat

parent_dir = os.path.abspath(os.path.join(os.path.dirname('../NEUROEVOLUTION/gpolnel'), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from NEUROEVOLUTION.utils import seed, scale, clipp, cv_logger_init, cv_logger_append

import csv
import numpy as np   




def neat_train(X_train, y_train, X_val, y_val, config_path, n_iter, log_path):
    
    def eval_genomes(genomes, config):
        '''
        The function used by NEAT-Python to evaluate the fitness of the genomes.
        -> It has to have the two first arguments equals to the genomes and config objects.
        -> It has to update the `fitness` attribute of the genome.
        '''
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = eval_rmse(net, X_train, y_train)
            genome.fitness_val = eval_rmse(net, X_val, y_val)
    
    config_file = config_path
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, filename=config_file)
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, n_iter)
    
    
    iterations = range(0,n_iter)
    best = [c.fitness*-1 for c in stats.most_fit_genomes]
    best_val = [c.fitness_val *-1 for c in stats.most_fit_genomes]
    
    
    
    
    for i in range(len(iterations)):
        with open(log_path, 'a', newline='\n') as csvfile:
            w = csv.writer(csvfile, delimiter=';')
            w.writerow([iterations[i]]+[best[i]]+[best_val[i]])

    return winner


def neat_cross_validation( X_train, y_train, config_path, kf, n_iter,  train_log_path, cv_log_path, id=None):#v_log_path,
    
    cv_logger_init(cv_log_path)
    results = []
    with open(train_log_path, 'w', newline='\n') as csvfile:
            w = csv.writer(csvfile, delimiter=';')
            w.writerow(['generation']+['train_score']+['val_score'])
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        
        
        X_train_cross, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cross, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        X_train_cross, X_val = clipp(X_train_cross, X_val)
        X_train_cross, X_val = scale(X_train_cross, X_val)
        X_train_cross, X_val = X_train_cross.to_numpy(), X_val.to_numpy()
        
        # X_train_cross = torch.tensor(X_train_cross.values, dtype=torch.float32)
        # X_val = torch.tensor(X_val.values, dtype=torch.float32)
        # y_train_cross = torch.tensor(y_train_cross.values, dtype=torch.float32).reshape(-1, 1)
        # y_val = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
        
        
        model = neat_train(X_train_cross, y_train_cross, X_val, y_val, config_path, log_path = train_log_path, n_iter=n_iter)
        
        score = model.fitness_val *-1

        cv_logger_append(cv_log_path, id, fold, score, 'repr')

        results.append(score)
        
    avg_score = np.mean(results)
    print(avg_score)
    return avg_score


def eval_rmse(net, X, y):
    '''
    Auxiliary funciton to evaluate the RMSE.
    '''
    fit = 0.
    for xi, xo in zip(X, y):
        output = net.activate(xi)
        fit += (output[0] - xo)**2
    # RMSE
    return -(fit/y.shape[0])**.5



