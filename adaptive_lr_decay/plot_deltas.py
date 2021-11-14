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

fedavg_mbsgd = '/ocean/projects/iri180031p/houc/multistage/hypothetical_GHOST/results/EMNIST_ghost/experiment.metrics.csv'
additional = '/ocean/projects/iri180031p/houc/multistage/hypothetical_GHOST/results/EMNIST_ghost_additional/experiment.metrics.csv'
for smooth in [0,1,5,10,15,20,25,30]:
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)

    best_file = fedavg_mbsgd
    df = pd.read_csv(best_file, sep=',')
    df_additional = pd.read_csv(additional, sep=',')
    print(df['train_eval_mbsgd/loss'])
    for col in ['prev_model_train_eval/loss', 'train_eval_mbsgd/loss', 'train_eval/loss', 'train_eval_fedavg/loss']:
        df.loc[df[col].isna(),col] = df_additional[df[col].isna()][col]
    
    previous_loss = df['prev_model_train_eval/loss'].dropna().ewm(com=smooth).mean().to_numpy()
    model_loss = df['train_eval/loss'].dropna().ewm(com=smooth).mean().to_numpy()
    mbsgd_loss = df['train_eval_mbsgd/loss'].dropna().ewm(com=smooth).mean().to_numpy()
    fedavg_loss = df['train_eval_fedavg/loss'].dropna().ewm(com=smooth).mean().to_numpy()
    print(df['train_eval_mbsgd/loss'])
    x = np.where(df['train_eval/loss'].notna().to_numpy().squeeze())[0]
    ax.plot(x, previous_loss - fedavg_loss, '-o', label='FedAvg', linewidth = 5.0, markersize=15.0)
    ax.plot(x, previous_loss - mbsgd_loss, '-o', label='SGD', linewidth = 5.0, markersize=15.0)
    #ax.plot(x, previous_loss - model_loss, '-.P', label='Model', markevery=[7], linewidth = 5.0, markersize=30.0)
    plt.axvline(x=350, linewidth = 5.0, c='red')

    plt.xlabel('Rounds', fontsize=30)
    plt.ylabel('Progress', fontsize=30)
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f'/jet/home/houc/multistage/federated/adaptive_lr_decay/deltas/delta-{smooth}.png', dpi=300)
    plt.close()
    print('\n\n')







