import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os


lr_grid = [0.1, 0.2, 0.30000000000000004, 0.4, 0.5]
switch_grid = [0.1, 0.30000000000000004, 0.5, 0.7000000000000001, 0.9]
multistage_first_grid = [0.01, 0.023713737056616554, 0.05623413251903491, 0.1333521432163324, 0.31622776601683794]
# eval/sparse_categorical_accuracy, eval/loss
#metric = 'train_eval/loss'

dir_path = '/ocean/projects/iri180031p/houc/multistage/grid_out/results/'

fedavg_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/results/results/EMNIST_{lr}_switch_{switch}'
scaffold_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/results/results/EMNIST_{lr}_switch_{switch}_control'
fedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/results/results/EMNIST_{lr}_switch_200'
scaffold_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/results/results/EMNIST_{lr}_switch_200_control'
Mfedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/grid_out/results/EMNISTMULTI_sample_10_lr_{lr}_switch_{switch}_control_0_allowswap_1'
Mscaffold_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/grid_out/results/EMNISTMULTI_sample_10_lr_{lr}_switch_{switch}_control_1_allowswap_1'

def plot_metric(metric, fedavg_mbsgd_fn, scaffold_mbsgd_fn, fedavg_fn,
                scaffold_fn, Mfedavg_fn, Mscaffold_fn):
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
                    #print(filename)
                    continue
                df = pd.read_csv(filename, sep=',')
                df_len = len(df[metric])
                final_y = metric_fn(df[metric])[df_len - 5]
                #print(final_y, best_val)
                if final_y < best_val:
                    best_val = final_y
                    best_file = filename
        #print(best_file)
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

    best_file = find_best_in_grid(lr_grid, multistage_first_grid, Mscaffold_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-SCAFFOLD', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-SCAFFOLD')

    best_file = find_best_in_grid(lr_grid, multistage_first_grid, Mfedavg_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])
    print('M-FedAvg', min(y))
    x = range(len(y))
    plt.plot(x,y, label='M-FedAvg')



    plt.xlabel('Rounds')
    plt.ylabel(metric) 
    plt.legend()
    plt.yscale('log')
    save_metric_name = metric.replace('/', '_')
    plt.tight_layout()
    plt.savefig(f'/jet/home/houc/multistage/federated/adaptive_lr_decay/sample10_figs/{save_metric_name}.png', dpi=300)
    plt.close()

metric = 'eval/sparse_categorical_accuracy'
for metric in ['eval/sparse_categorical_accuracy', 'eval/loss', 'train_eval/loss', 'train_eval/sparse_categorical_accuracy']:
    plot_metric(metric, fedavg_mbsgd_fn, scaffold_mbsgd_fn, fedavg_fn,
                scaffold_fn, Mfedavg_fn, Mscaffold_fn)



