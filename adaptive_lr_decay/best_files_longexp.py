import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os


lr_grid = [0.1, 0.15, 0.2, 0.25, 0.3]
switch_grid = [0.1, 0.30000000000000004, 0.5, 0.7000000000000001, 0.9]
multistage_first_grid = [0.001, 0.0015, 0.002, 0.0025, 0.0032500000000000003]
# eval/sparse_categorical_accuracy, eval/loss
#metric = 'train_eval/loss'
dir_path = '/ocean/projects/iri180031p/houc/multistage/grid_out/results/'

fedavg_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/long_exp/results/EMNIST_{lr}_switch_{switch}_control_0'
scaffold_mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/long_exp/results/EMNIST_{lr}_switch_{switch}_control_1'
fedavg_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/long_exp/results/EMNIST_{lr}_switch_200_control_0'
scaffold_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/long_exp/results/EMNIST_{lr}_switch_200_control_1'
mbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/long_exp/results/EMNIST_Minibatch_{lr}_switch_200_multi_0'
Mmbsgd_fn = lambda lr, switch: f'/ocean/projects/iri180031p/houc/multistage/long_exp/results/EMNIST_Minibatch_{lr}_switch_{switch}_multi_1'
Mfedavg_Mmbsgd_fn = lambda lr, switch, swap: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/EMNISTMULTI_3_{lr}_switch_{switch}_control_0_swapround_{swap}'
Mscaffold_Mmbsgd_fn = lambda lr, switch, swap: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/EMNISTMULTI_3_{lr}_switch_{switch}_control_1_swapround_{swap}'
Mfedavg_fn = lambda lr, switch, swap: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/EMNISTMULTI_3_{lr}_switch_{switch}_control_0_swapround_{swap}'
Mscaffold_fn = lambda lr, switch, swap: f'/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/EMNISTMULTI_3_{lr}_switch_{switch}_control_1_swapround_{swap}'

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

    def find_best_in_grid(lr_grid, switch_grid, exp_name_fn, swapgrid=[]):
        best_val = 1e10
        best_file = ''
        best_lr = -1
        switch_round = -1
        swap_round = -1
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
                            best_lr = lr
                            switch_round = switch
                            swap_round = swap
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
                        best_lr = lr
                        switch_round = switch
        print(best_file)
        return best_file, best_lr, switch_round, swap_round
        
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)
    

    

    best_file, _, switch, swap = find_best_in_grid(lr_grid, [200], mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('Minibatch SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers = getmarkers('twostage', switch, 500)
    ax.plot(x,y, label='SGD',linewidth = 5.0)

    best_file, _, switch, swap = find_best_in_grid(lr_grid, [200], fedavg_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('FedAvg', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers = getmarkers('twostage', switch, 500)
    ax.plot(x,y, label='FedAvg',linewidth = 5.0)

    best_file, _, switch, swap = find_best_in_grid(lr_grid, [200], scaffold_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('SCAFFOLD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers = getmarkers('twostage', switch, 500)
    ax.plot(x,y, label='SCAFFOLD', linewidth = 5.0)

    best_file, _, switch, swap = find_best_in_grid([0.1, 0.2, 0.30000000000000004, 0.4, 0.5], [0.01, 0.023713737056616554, 0.05623413251903491, 0.1333521432163324, 0.31622776601683794], Mmbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-Minibatch SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers = getmarkers('simple', switch, 500)
    ax.plot(x,y, '-D', label='M-SGD', markevery=markers, linewidth = 5.0, markersize=30.0)

    best_file, _, switch, swap = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.275, 0.45], Mfedavg_fn, swapgrid=[200])
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-FedAvg', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers = getmarkers('simple', switch, 500)
    ax.plot(x,y, '-D',label='M-FedAvg', markevery=markers, linewidth = 5.0, markersize=30.0)

    best_file, _, switch, swap = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.275, 0.45], Mscaffold_fn, swapgrid=[200])
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-SCAFFOLD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers = getmarkers('simple', switch, 500)
    ax.plot(x,y, '-D',label='M-SCAFFOLD', markevery=markers, linewidth = 5.0, markersize=30.0)

    best_file, _, switch, swap = find_best_in_grid(lr_grid, switch_grid, fedavg_mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('FedAvg->Minibatch SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10) 
    markers = getmarkers('twostage', switch, 500)
    ax.plot(x,y, '-.P', label='FedAvg->SGD', markevery=markers, linewidth = 5.0, markersize=30.0)

    best_file, _, switch, swap = find_best_in_grid(lr_grid, switch_grid, scaffold_mbsgd_fn)
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('SCAFFOLD->SGD', np.mean(y[45:]))
    x = range(0, len(y)*10, 10) 
    markers = getmarkers('twostage', switch, 500)
    ax.plot(x,y, '-.P', markevery=markers, linewidth = 5.0, markersize=30.0, label='SCAFFOLD->SGD')

    

    best_file, _, switch, swap = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.275, 0.45], Mfedavg_Mmbsgd_fn, swapgrid=[0.5, 0.7, 0.9])
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-FedAvg->M-Mbsgd', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers, switch_mark = getmarkers('local', switch, 500, swaplen=swap)
    ax.plot(x,y, '-.co', label='M-(FedAvg->SGD)',markevery=markers, linewidth = 5.0, markersize=30.0)
    ax.plot(x,y, '-.cP', markevery=switch_mark, linewidth = 5.0, markersize=30.0)

    best_file, _, switch, swap = find_best_in_grid([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.275, 0.45], Mscaffold_Mmbsgd_fn, swapgrid=[0.5, 0.7, 0.9])
    df = pd.read_csv(best_file, sep=',')
    y = metric_fn(df[metric])[::10]
    print('M-SCAFFOLD->M-Mbsgd', np.mean(y[45:]))
    x = range(0, len(y)*10, 10)
    markers, switch_mark = getmarkers('local', switch, 500, swaplen=swap)
    ax.plot(x,y, '-.mo', label='M-(SCAFFOLD->SGD)',markevery=markers, linewidth = 5.0, markersize=30.0)
    ax.plot(x,y, '-.mP', markevery=switch_mark, linewidth = 5.0, markersize=30.0)

    



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
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.yscale('log')
    plt.title('EMNIST', fontdict = {"fontsize":30})
    save_metric_name = metric.replace('/', '_')
    plt.tight_layout()
    plt.savefig(f'/jet/home/houc/multistage/federated/adaptive_lr_decay/longexp_figs/{save_metric_name}.png', dpi=300)
    plt.close()
    print('\n\n')

metric = 'eval/sparse_categorical_accuracy'
for metric in ['eval/sparse_categorical_accuracy', 'eval/loss', 'train_eval/loss', 'train_eval/sparse_categorical_accuracy']:
    plot_metric(metric, fedavg_mbsgd_fn, scaffold_mbsgd_fn, fedavg_fn, scaffold_fn, 
                mbsgd_fn, Mmbsgd_fn,
                Mfedavg_fn, Mscaffold_fn, Mfedavg_Mmbsgd_fn, Mscaffold_Mmbsgd_fn)



