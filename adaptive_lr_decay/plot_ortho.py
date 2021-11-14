import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os
from scipy.signal import savgol_filter



lr_grid = [0.1, 0.15, 0.2, 0.25, 0.3]
switch_grid = [0.1, 0.30000000000000004, 0.5, 0.7000000000000001, 0.9]
multistage_first_grid = [0.001, 0.0015, 0.002, 0.0025, 0.0032500000000000003]
# eval/sparse_categorical_accuracy, eval/loss
#metric = 'train_eval/loss'

dir_path = '/ocean/projects/iri180031p/houc/multistage/grid_out/results/'

fedavg_mbsgd = '/ocean/projects/iri180031p/houc/multistage/orthogonal/results/EMNIST_0.1_switch_0.7_control_0/experiment.metrics.csv'
fedavg = '/ocean/projects/iri180031p/houc/multistage/orthogonal/results/EMNIST_0.1_switch_200_control_0/experiment.metrics.csv'
mbsgd = '/ocean/projects/iri180031p/houc/multistage/orthogonal/results/EMNIST_Minibatch_0.2_switch_200_multi_0/experiment.metrics.csv'
def plot_metric(metric, 
                fedavg_mbsgd, fedavg, mbsgd):
    print('Plot', metric)
    def build_metric_fn(metric):
        def metric_fn(y):
            if metric.split('/')[-1] == 'sparse_categorical_accuracy':
                return 1 - y
            else:
                return y
        return metric_fn

    metric_fn = build_metric_fn(metric)
    def build_filter_fn(metric):
        def filter_fn(y):
            if metric.split('/')[-1] == 'sgd_gradient_norm':
                print('filter triggered')
                df = pd.DataFrame(y)
                return df.ewm(com=20).mean()
            else:
                return y
        return filter_fn
    filter_fn = build_filter_fn(metric)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)
    
    best_file = fedavg
    df = pd.read_csv(best_file, sep=',')
    y = filter_fn(metric_fn(df[metric])[::10])
    print('FedAvg', min(y))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, label='FedAvg', linewidth = 5.0)


    best_file = mbsgd
    df = pd.read_csv(best_file, sep=',')
    y = filter_fn(metric_fn(df[metric])[::10])
    print('Minibatch SGD', min(y))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, label='SGD', linewidth = 5.0)

    best_file = fedavg_mbsgd
    df = pd.read_csv(best_file, sep=',')
    y = filter_fn(metric_fn(df[metric])[::10])
    print('FedAvg->Minibatch SGD', min(y))
    x = range(0, len(y)*10, 10)
    #print(y)
    ax.plot(x,y, '-.P', label='FedAvg->SGD', markevery=[35], linewidth = 5.0, markersize=30.0)


    plt.xlabel('Rounds', fontsize=30)
    if metric == 'train/sgd_gradient_norm':
        plt.ylabel('Stochastic Gradient Norm of Server Model', fontsize=30) 
    if metric == 'train/average_cosine_similarity':
        plt.ylabel('Average Pairwise Cosine Similarity', fontsize=30)
    if metric == 'train/server_update_norm':
        plt.ylabel('Server Update Norm', fontsize=30)    
    plt.legend()
    if metric in ['eval/sparse_categorical_accuracy', 'eval/loss', 'train_eval/loss', 
                'train_eval/sparse_categorical_accuracy']:
        plt.yscale('log')
    save_metric_name = metric.replace('/', '_')
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f'/jet/home/houc/multistage/federated/adaptive_lr_decay/orthogonal_figs/{save_metric_name}.png', dpi=300)
    plt.close()
    print('\n\n')

metric = 'eval/sparse_categorical_accuracy'
for metric in [ 'train/average_cosine_similarity',
                'train/server_update_norm',
                'train/sgd_gradient_norm']:
    plot_metric(metric, fedavg_mbsgd, fedavg, mbsgd)



