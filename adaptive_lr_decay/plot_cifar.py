import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os

from pandas.core.indexes import multi


lr_grid = [0.05, 0.1, 0.15, 0.2, 0.25]
switch_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
multistage_first_grid = [0.002, 0.0032, 0.0045, 0.0058, 0.007]
# eval/sparse_categorical_accuracy, eval/loss
#metric = 'train_eval/loss'

dir_path = '/ocean/projects/iri180031p/houc/multistage/grid_out/results/'

fedavg_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFAR_{lr}_switch_{switch}_control_0'
fedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFAR_{lr}_switch_200_control_0'
mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFAR_Minibatch_{lr}_switch_200_multi_0'
Mmbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFAR_Minibatch_{lr}_switch_{switch}_multi_1'
Mfedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFARMULTI_{lr}_switch_{switch}_control_0_allowswap_0'
Mfedavg_Mmbsgd_fn = lambda lr, switch, swapround: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFARMULTI_3_{lr}_switch_{switch}_control_0_swapround_{swapround}'
def local_schedule(lr, s, rounds_passed, stageone, R, clients, GDA_flag, opt):
    begin_client_lr = lr 
    begin_server_lr = lr
    curr_stage_length = int(stageone * R) * 2 ** s
    eta_E = 0
    if rounds_passed > curr_stage_length:
        currstage = s + 1
        currrounds = 0
    else:
        currstage = s
        currrounds = rounds_passed + 1

    if GDA_flag:
        client_lr = 0
        server_lr = begin_server_lr * 2. **(-s) / clients
    else:
        client_lr = begin_client_lr * 2. **(-s)
        server_lr = begin_server_lr * 2. **(-s)

    if client_lr < lr /clients and GDA_flag == False:
        client_lr = 0
        currstage = 0
        GDA_flag = True
    if GDA_flag == True and opt == 1:
        eta_E = server_lr
    return currstage, currrounds, client_lr, server_lr, GDA_flag, eta_E
def simple_schedule(lr, s, rounds_passed, stageone, R):
    curr_stage_length = int(stageone * R) * 2 ** s
    if rounds_passed > curr_stage_length:
        currstage = s + 1
        currrounds = 0
    else:
        currstage = s
        currrounds = rounds_passed + 1
    return currstage, currrounds, lr * 2**(-s)
def getmarkers(stagetype, stagelen, R, swaplen=None):
    if stagetype == 'twostage':
        return [int(stagelen*R/10)]
    elif stagetype == 'local':
        s = 0.
        rounds_in_stage = 0.
        swapped = False
        roundmarkers = []
        switchmarker = []
        for i in range(R):
            s_ = s
            curr_stage_length = stagelen * 2. ** s * R
            if rounds_in_stage >= int(curr_stage_length):
                s += 1.
                rounds_in_stage = 0
            else:
                rounds_in_stage += 1
            
            if i >= swaplen*R and not swapped:
                swapped = True
                s = 0.
                switchmarker.append(int(i/10.))
            if s_ != s:
                roundmarkers.append(int(i/10.))
            s = s_
        return roundmarkers, switchmarker
    else:
        round_markers = []
        s = 0 
        lr = 0.1
        rounds_passed = 0
        stageone = stagelen
        clients = 200
        GDA_flag = False 
        opt = False
        for i in range(R):
            s_, rounds_passed, _ = simple_schedule(lr, s, rounds_passed, stageone, R)
            if s_ != s:
                round_markers.append(int(i/10.))
            s = s_
        return round_markers
def plot_metric(metric, 
                fedavg_mbsgd_fn, fedavg_fn, mbsgd_fn, Mmbsgd_fn, Mfedavg_Mmbsgd_fn, Mfedavg_fn):
    print('Plot', metric)
    def build_metric_fn(metric):
        def metric_fn(y):
            if metric.split('/')[-1] == 'sparse_categorical_accuracy':
                return 1 - y
            else:
                return y
        return metric_fn

    metric_fn = build_metric_fn(metric)

    def find_best_in_grid(lr_grid, switch_grid, exp_name_fn, swapgrid=[]):
        best_val = 1e10
        best_file = ''
        for lr in lr_grid:
            for switch in switch_grid:
                if swapgrid:
                    for swap in swapgrid:
                        exp_name = exp_name_fn(lr, switch,swap)
                        filename = os.path.join(exp_name, 'experiment.metrics.csv')
                        if not os.path.exists(filename):
                            continue
                        df = pd.read_csv(filename, sep=',')
                        df_len = len(df[metric])
                        conv_y = metric_fn(df[metric])
                        final_y = conv_y.dropna().to_numpy()
                        final_y = np.mean(final_y[45:])
                        if final_y < best_val:
                            best_val = final_y
                            best_file = filename
                else:
                    exp_name = exp_name_fn(lr, switch)
                    filename = os.path.join(exp_name, 'experiment.metrics.csv')
                    if not os.path.exists(filename):
                        continue
                    df = pd.read_csv(filename, sep=',')
                    df_len = len(df[metric])
                    conv_y = metric_fn(df[metric])
                    final_y = conv_y.dropna().to_numpy()
                    final_y = np.mean(final_y[45:])
                    if final_y < best_val:
                        best_val = final_y
                        best_file = filename
        print(best_file)
        return best_file
        
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)
    
    best_file = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [200], mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, label='SGD', linewidth = 5.0, )
    
    best_file = find_best_in_grid(lr_grid, [200], fedavg_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('FedAvg', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, label='FedAvg', linewidth = 5.0)


    

    best_file = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.01, 0.0237, 0.0562, 0.1334, 0.3162], Mmbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, '-o', markevery=getmarkers('simple', 0.1334,500), label='M-SGD', linewidth = 5.0, markersize=30.0)

    #best_file = '/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFARMULTI_0.1_switch_0.0045_control_0_allowswap_1/experiment.metrics.csv'
    best_file = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.275, 0.45], Mfedavg_Mmbsgd_fn, swapgrid=[0.5, 0.7, 0.9])
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-(FedAvg->SGD)', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    #print(getmarkers('local', 0.45,500, swaplen=0.7))
    switchmarkers, swapmarkers = getmarkers('local', 0.45,500, swaplen=0.7)
    ax.plot(x,y, '-co', markevery=switchmarkers, label='M-(FedAvg->SGD)', linewidth = 5.0, markersize=30.0)
    ax.plot(x,y, '-cP', markevery=swapmarkers, linewidth = 5.0, markersize=30.0)

    best_file = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.275, 0.45], Mfedavg_Mmbsgd_fn, swapgrid=[200])
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-FedAvg', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, '-o', markevery=getmarkers('simple', 0.275,500), label='M-FedAvg', linewidth = 5.0, markersize=30.0)

    best_file = find_best_in_grid(lr_grid, switch_grid, fedavg_mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('FedAvg->SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    ax.plot(x,y, '-.P', markevery=[35], label='FedAvg->SGD', linewidth = 5.0, markersize=30.0)

    

    plt.xlabel('Rounds', fontsize = 30)
    if metric == 'eval/sparse_categorical_accuracy':
        plt.ylabel('Test Error', fontsize = 30) 
    elif metric == 'train_eval/sparse_categorical_accuracy':
        plt.ylabel('Train Error', fontsize = 30) 
    elif metric == 'train_eval/loss':
        plt.ylabel('Train Loss', fontsize = 30) 
    elif metric == 'eval/loss':
        plt.ylabel('Test Loss', fontsize = 30) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)
    plt.title('CIFAR-100', fontdict={'fontsize':30})
    #plt.yscale('log')
    save_metric_name = metric.replace('/', '_')
    plt.tight_layout()
    
    plt.savefig(f'/jet/home/houc/multistage/federated/adaptive_lr_decay/cifar_figs/{save_metric_name}.png', dpi=300)
    plt.close()
    print('\n\n')

metric = 'eval/sparse_categorical_accuracy'
for metric in ['eval/sparse_categorical_accuracy', 'eval/loss', 'train_eval/loss', 'train_eval/sparse_categorical_accuracy']:
    plot_metric(metric, fedavg_mbsgd_fn, fedavg_fn,
                mbsgd_fn, Mmbsgd_fn, Mfedavg_Mmbsgd_fn, Mfedavg_fn)



