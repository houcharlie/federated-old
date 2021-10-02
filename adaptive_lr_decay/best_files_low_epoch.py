import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os


lr_grid = [0.1, 0.15, 0.2, 0.25, 0.3]
switch_grid = [0.1, 0.30000000000000004, 0.5, 0.7000000000000001, 0.9]
multistage_first_grid = [0.01, 0.023713737056616554, 0.05623413251903491, 0.1333521432163324, 0.31622776601683794]
# eval/sparse_categorical_accuracy, eval/loss
#metric = 'train_eval/loss'

dir_path = '/ocean/projects/iri180031p/houc/multistage/grid_out/results/'

fedavg_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNIST_{lr}_switch_{switch}_control_0'
scaffold_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNIST_{lr}_switch_{switch}_control_1'
fedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNIST_{lr}_switch_200_control_0'
scaffold_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNIST_{lr}_switch_200_control_1'
mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNIST_Minibatch_{lr}_switch_200_multi_0'
Mmbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNIST_Minibatch_{lr}_switch_{switch}_multi_1'
Mfedavg_Mmbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNISTMULTI_{lr}_switch_{switch}_control_0_allowswap_1'
Mscaffold_Mmbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNISTMULTI_{lr}_switch_{switch}_control_1_allowswap_1'
Mfedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNISTMULTI_{lr}_switch_{switch}_control_0_allowswap_0'
Mscaffold_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/low_epochs/results/EMNISTMULTI_{lr}_switch_{switch}_control_1_allowswap_0'

def plot_metric(metric, 
                fedavg_mbsgd_fn, scaffold_mbsgd_fn, fedavg_fn, scaffold_fn, 
                mbsgd_fn, Mmbsgd_fn,
                Mfedavg_fn, Mscaffold_fn, Mfedavg_Mmbsgd_fn, Mscaffold_Mmbsgd_fn):
    print('Plot', metric)
    def build_metric_fn(metric):
        def metric_fn(y):
            if metric.split('/')[-1] == 'sparse_categorical_accuracy':
                return 1 - y
            else:
                return y
        return metric_fn

    metric_fn = build_metric_fn(metric)

    def find_best_in_grid(lr_grid, switch_grid, exp_name_fn):
        best_val = 1e10
        best_file = ''
        for lr in lr_grid:
            for switch in switch_grid:
                exp_name = exp_name_fn(lr, switch)
                filename = os.path.join(exp_name, 'experiment.metrics.csv')
                if not os.path.exists(filename):
                    continue
                df = pd.read_csv(filename, sep=',')
                df_len = len(df[metric])
                final_y = metric_fn(df[metric])[df_len - 5]
                if final_y < best_val:
                    best_val = final_y
                    best_file = filename
        print(best_file)
        return best_file
        

    best_file = find_best_in_grid(lr_grid, switch_grid, fedavg_mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('FedAvg->Minibatch SGD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='FedAvg->Minibatch SGD')

    best_file = find_best_in_grid(lr_grid, switch_grid, scaffold_mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('SCAFFOLD->Minibatch SGD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='SCAFFOLD->Minibatch SGD')

    best_file = find_best_in_grid(lr_grid, [200], fedavg_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('FedAvg', min(y))
    x = range(len(y))
    plt.plot(x,y, label='FedAvg')

    best_file = find_best_in_grid(lr_grid, [200], scaffold_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('SCAFFOLD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='SCAFFOLD')

    best_file = find_best_in_grid(lr_grid, [200], mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('Minibatch SGD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='Minibatch SGD')

    best_file = find_best_in_grid(lr_grid, multistage_first_grid, Mmbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-Minibatch SGD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-Minibatch SGD')

    best_file = find_best_in_grid(lr_grid, multistage_first_grid, Mfedavg_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-FedAvg', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-FedAvg')

    best_file = find_best_in_grid(lr_grid, multistage_first_grid, Mscaffold_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-SCAFFOLD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-SCAFFOLD')

    best_file = find_best_in_grid(lr_grid, [0.01, 0.023713737056616554, 0.05623413251903491], Mfedavg_Mmbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-FedAvg->M-Mbsgd', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-FedAvg -> M-Mbsgd')

    best_file = find_best_in_grid(lr_grid, [0.01, 0.023713737056616554, 0.05623413251903491], Mscaffold_Mmbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-SCAFFOLD->M-Mbsgd', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-SCAFFOLD -> M-Mbsgd')

    



    plt.xlabel('Rounds')
    plt.ylabel(metric) 
    plt.legend()
    plt.yscale('log')
    save_metric_name = metric.replace('/', '_')
    plt.tight_layout()
    plt.savefig(f'/jet/home/houc/multistage/federated/adaptive_lr_decay/low_epoch_figs/{save_metric_name}.png', dpi=300)
    plt.close()

metric = 'eval/sparse_categorical_accuracy'
for metric in ['eval/sparse_categorical_accuracy', 'eval/loss', 'train_eval/loss', 'train_eval/sparse_categorical_accuracy']:
    plot_metric(metric, fedavg_mbsgd_fn, scaffold_mbsgd_fn, fedavg_fn, scaffold_fn, 
                mbsgd_fn, Mmbsgd_fn,
                Mfedavg_fn, Mscaffold_fn, Mfedavg_Mmbsgd_fn, Mscaffold_Mmbsgd_fn)



