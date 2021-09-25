import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import os
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def wgrad(X, Y, w, mu):
    Y =  Y.reshape(-1,1)
    z = X @ w
    N = X.shape[0]
    h = sigmoid(z)
    return (X.T @ (h - Y))/N + mu * w
def loss(X, Y, w, mu):
    y =  Y.reshape(-1,1)
    z = X @ w
    h = sigmoid(z)
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (mu/2)*np.linalg.norm(mu)**2
def make_bars(vals):
    vals = np.array(vals)
    meantrials = np.mean(vals,axis = 0)
    errorbar_low = np.abs(np.quantile(vals, 0.25, axis = 0) - meantrials)
    errorbar_high = np.abs(np.quantile(vals, 0.75, axis = 0) - meantrials)
    errorbars = np.vstack((errorbar_low, errorbar_high))
    return vals, meantrials, errorbars

def het(w_grad_eval, X_fed, Y_fed, w, mu, clients):
    sumdiffs = 0
    for i in range(clients):
        w_grad = wgrad(X_fed[i,:,:], Y_fed[i,:], w, mu).reshape(-1,1)
        sumdiffs += np.linalg.norm(w_grad_eval - w_grad)**2
    return sumdiffs/float(clients)


def local_schedule(lr, s, rounds_passed, stageone, R, K, GDA_flag, opt):
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
        server_lr = begin_server_lr * 2. **(-s) / K
    else:
        client_lr = begin_client_lr * 2. **(-s)
        server_lr = begin_server_lr * 2. **(-s)

    if client_lr < lr /K and GDA_flag == False:
        client_lr = 0
        currstage = 0
        GDA_flag = True
    if GDA_flag == True and opt == 1:
        eta_E = server_lr
        
    return currstage, currrounds, client_lr, server_lr, GDA_flag, eta_E
def switch_schedule(lr, r, R, stageone, K, opt):
    if r > R * stageone:
        if opt == 1:
            return lr/K, 0, lr/K
        else:
            return lr/K, 0, 0
    else:
        return lr, lr, 0
def simple_schedule(lr, s, rounds_passed, stageone, R):
    curr_stage_length = int(stageone * R) * 2 ** s
    if rounds_passed > curr_stage_length:
        currstage = s + 1
        currrounds = 0
    else:
        currstage = s
        currrounds = rounds_passed + 1
        
    return currstage, currrounds, lr * 2**(-s)

def run_GDA(args):
     ## Load data
    if os.path.exists(os.path.join(dir_path,'x.npy')):
        print('Loading data')
        X = np.load(os.path.join(dir_path, 'x.npy'), allow_pickle=True)
        Y = np.load(os.path.join(dir_path, 'y.npy'), allow_pickle=True)
    else:
        print('Downloading and saving data')
        mnist = fetch_openml('mnist_784')
        X = mnist['data'].to_numpy()/255.
        Y = mnist['target'].to_numpy()
        np.save('x.npy', X)
        np.save('y.npy', Y)
        print('Done downloading and saving data')
    # half of the classes are 0, and other half is 1
    print("Running Extragrad")
    vfunc = np.vectorize(lambda x: int(x))
    Y = vfunc(Y)
    n_0 = 500
    n_1 = 500
    d = X.shape[1]
    R = args.R
    clients = 5
    X_fed = np.zeros((clients, n_0 + n_1, d))
    Y_fed = np.zeros((clients, n_0 + n_1))
    # split into clients
    for i in range(clients):
        X_fed[i, :n_0, :] = X[Y==2*i][:n_0,:]
        X_fed[i, n_0:, :] = X[Y==2*i + 1][:n_1,:]
        # the even labels are 0s, odds are 1
        Y_fed[i,:n_0] = 0
        Y_fed[i,n_0:] = 1
    ones = n_1 * clients 
    zeros = n_0 * clients 
    p = float(ones)/float(ones + zeros)
    mu = 0.1
    print('Strong convexity: ', mu)
    X_full = X_fed.reshape(-1,784)
    Y_full = Y_fed.reshape(-1,1).squeeze()
    X = X_full 
    Y = Y_full
    if args.batch == 1:
        trials = 1 
    else:
        trials = 1000
    batch = min(int(args.batch * (n_1 + n_0) * args.K * clients), int((n_1 + n_0) * clients))
    trialvals = []
    lr = args.lr
    eta = lr
    for t in range(trials):
        w = np.zeros((d,1))
        s = 0
        rounds_passed = 0
        grad_norms = []
        for r in range(R):
            idx = np.random.choice((n_1 + n_0) * clients, size = batch, replace = False)
            w_grad = wgrad(X[idx,:], Y[idx], w, mu).reshape(-1,1)
            w = w - eta * w_grad

            grad_norm = np.linalg.norm(w_grad)**2
            if r % 1 == 0:
                w_grad_eval = wgrad(X_full, Y_full, w, mu)
                loss_eval = loss(X_full,Y_full,w,mu)
                grad_norm = np.linalg.norm(w_grad_eval)**2
                print('Trial {0} Round {1} Eta {3}: Grad {2} Loss {4}'.format(t, r, grad_norm, eta, loss_eval))
                grad_norms.append(grad_norm)
            if args.multistage == 1:
                s, rounds_passed, eta = simple_schedule(lr, s, rounds_passed, args.stageone, R)
        trialvals.append(grad_norms)

    gradvals, gradmeans, gradbars = make_bars(trialvals)

    dirpath = os.path.join(dir_path, 'logistic/rounds{3}/batch{0}/GDA/stage{1}/stageone{2}'.format(args.batch, args.multistage, args.stageone,args.R))
    expname = 'lr{0}'.format(args.lr)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath,  exist_ok=True)  
    plt.errorbar(range(gradmeans.shape[0]), gradmeans, yerr = gradbars)
    plt.xlabel('Rounds')
    plt.ylabel('Grad Norm')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(dir_path, os.path.join(dirpath, expname + '.png')))
    np.save(os.path.join(dir_path, os.path.join(dirpath, expname)), gradvals)


def run_extragrad(args):
     ## Load data
    if os.path.exists(os.path.join(dir_path,'x.npy')):
        print('Loading data')
        X = np.load(os.path.join(dir_path, 'x.npy'), allow_pickle=True)
        Y = np.load(os.path.join(dir_path, 'y.npy'), allow_pickle=True)
    else:
        print('Downloading and saving data')
        mnist = fetch_openml('mnist_784')
        X = mnist['data'].to_numpy()/255.
        Y = mnist['target'].to_numpy()
        np.save('x.npy', X)
        np.save('y.npy', Y)
        print('Done downloading and saving data')
    # half of the classes are 0, and other half is 1
    print("Running AGD")
    vfunc = np.vectorize(lambda x: int(x))
    Y = vfunc(Y)
    n_0 = 500
    n_1 = 500
    d = X.shape[1]
    R = args.R
    clients = 5
    X_fed = np.zeros((clients, n_0 + n_1, d))
    Y_fed = np.zeros((clients, n_0 + n_1))
    # split into clients
    for i in range(clients):
        X_fed[i, :n_0, :] = X[Y==2*i][:n_0,:]
        X_fed[i, n_0:, :] = X[Y==2*i + 1][:n_1,:]
        # the even labels are 0s, odds are 1
        Y_fed[i,:n_0] = 0
        Y_fed[i,n_0:] = 1
    ones = n_1 * clients 
    zeros = n_0 * clients 
    p = float(ones)/float(ones + zeros)
    mu = 0.1
    print('Strong convexity: ', mu)
    X_full = X_fed.reshape(-1,784)
    Y_full = Y_fed.reshape(-1,1).squeeze()
    X = X_full 
    Y = Y_full
    if args.batch == 1:
        trials = 1 
    else:
        trials = 1000
    batch = min(int(args.batch * (n_1 + n_0) * args.K * clients), int((n_1 + n_0) * clients))
    print(batch)
    trialvals = []
    lr = args.lr
    eta = lr
    beta = (1-np.sqrt(eta * mu))/(1 + np.sqrt(eta * mu))
    for t in range(trials):
        w = np.zeros((d,1))
        s = 0
        rounds_passed = 0
        w_prev = np.zeros((d,1))
        grad_norms = []
        for r in range(R):
            idx = np.random.choice((n_1 + n_0) * clients, size = batch, replace = False)
            whalf = (1 + beta) * w - beta * w_prev
            w_grad = wgrad(X[idx,:], Y[idx], whalf, mu).reshape(-1,1)
            w_prev = w
            w = whalf - eta * w_grad
            grad_norm = np.linalg.norm(w_grad)**2
            if r % 1 == 0:
                w_grad_eval = wgrad(X_full, Y_full, w, mu)
                grad_norm = np.linalg.norm(w_grad_eval)**2
                loss_eval = loss(X_full,Y_full,w,mu)
                print('Trial {0} Round {1} Eta {3}: Grad {2} Loss {4} '.format(t, r, grad_norm, eta, loss_eval))
                grad_norms.append(grad_norm)
            if args.multistage == 1:
                s, rounds_passed, eta = simple_schedule(lr, s, rounds_passed, args.stageone, R)
        trialvals.append(grad_norms)
    dirpath = os.path.join(dir_path, 'logistic/rounds{3}/batch{0}/ACC/stage{1}/stageone{2}'.format(args.batch, args.multistage, args.stageone,args.R))

    gradvals, gradmeans, gradbars = make_bars(trialvals)

    
    expname = 'lr{0}'.format(args.lr)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath,  exist_ok=True)  
    plt.errorbar(range(gradmeans.shape[0]), gradmeans, yerr = gradbars)
    plt.xlabel('Rounds')
    plt.ylabel('Grad Norm')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(dir_path, os.path.join(dirpath, expname + '.png')))
    np.save(os.path.join(dir_path, os.path.join(dirpath, expname)), gradvals)



def grad_comp(n_0, n_1, w,  clients, d, K, batch, X_fed, Y_fed, X_full, Y_full, mu,args, eta):
    if eta > 0:
        w_grad_sum = np.zeros((d,1))

        w_cli = np.tile(w[np.newaxis,:,:], (clients, 1,1))

        ci_ws = np.zeros((clients,d,1))
        for i in range(clients):
            idx = np.random.choice(n_0 + n_1, size = min(K * batch, n_0 + n_1), replace = False)
            ci_ws[i,:,:] = wgrad(X_fed[i,idx,:], Y_fed[i,idx], w, mu).reshape(-1,1)
        c_w = np.mean(ci_ws, axis = 0)
        for i in range(clients):
            if args.control == 1:
                control_w = c_w - ci_ws[i,:,:]
            else:
                control_w = np.zeros_like(c_w)
            for k in range(K):
                idx = np.random.choice(n_0 + n_1, size = batch, replace = False)
                w_grad = wgrad(X_fed[i,idx,:], Y_fed[i,idx], w_cli[i,:,:], mu).reshape(-1,1)
                w_grad_sum += w_grad
                w_cli[i,:,:] = w_cli[i,:,:] - eta * (w_grad + control_w)
    else:
        batch = min(int(args.batch * (n_1 + n_0) * args.K * clients), int((n_1 + n_0) * clients))
        idx = np.random.choice((n_1 + n_0) * clients, size = batch, replace = False)
        w_grad = wgrad(X_full[idx,:], Y_full[idx], w, mu).reshape(-1,1)
        w_grad_sum = w_grad * K * clients


    return w_grad_sum, (w_grad)
def run_experiment(args):
    ## Load data
    if os.path.exists(os.path.join(dir_path,'x.npy')):
        print('Loading data')
        X = np.load(os.path.join(dir_path, 'x.npy'), allow_pickle=True)
        Y = np.load(os.path.join(dir_path, 'y.npy'), allow_pickle=True)
    else:
        print('Downloading and saving data')
        mnist = fetch_openml('mnist_784')
        X = mnist['data'].to_numpy()/255.
        Y = mnist['target'].to_numpy()
        np.save('x.npy', X)
        np.save('y.npy', Y)
        print('Done downloading and saving data')
    
    #X = np.load(os.path.join(dir_path, 'x-cifar.npy'), allow_pickle=True)
    #Y = np.load(os.path.join(dir_path, 'y-cifar.npy'), allow_pickle=True)
    #print('Got CIFAR')
    # half of the classes are 0, and other half is 1
    vfunc = np.vectorize(lambda x: int(x))
    Y = vfunc(Y)
    n_0 = 500
    n_1 = 500
    d = X.shape[1]
    K = args.K
    batch = int((n_0 + n_1) * args.batch)
    if args.batch == 1:
        trials = 1
    else:
        trials = 1000
    R = args.R
    clients = 5
    X_fed = np.zeros((clients, n_0 + n_1, d))
    Y_fed = np.zeros((clients, n_0 + n_1))


    # split into clients
    for i in range(clients):
        X_fed[i, :n_0, :] = X[Y==2*i][:n_0,:]
        X_fed[i, n_0:, :] = X[Y==2*i + 1][:n_1,:]
        # the even labels are 0s, odds are 1
        Y_fed[i,:n_0] = 0
        Y_fed[i,n_0:] = 1

        # shuffle the indices
        idx = np.random.permutation(n_0 + n_1)

        X_fed[i,idx,:] = X_fed[i,:,:]
        Y_fed[i,idx] = Y_fed[i,:]
    ones = n_1 * clients 
    zeros = n_0 * clients 
    p = float(ones)/float(ones + zeros)
    mu = 0.1
    print('Strong convexity: ', mu)
    X_full = X_fed.reshape(-1,d)
    Y_full = Y_fed.reshape(-1,1).squeeze()

    shufflepercent = args.shuffle_pct
    num_shuffle = int((n_0 + n_1)*shufflepercent)
    X_fed_shuffles = np.zeros((clients, num_shuffle, d))
    Y_fed_shuffles = np.zeros((clients, num_shuffle))
    for i in range(clients):
        X_fed_shuffles[i,:num_shuffle,:] = X_fed[i,:num_shuffle,:]
        Y_fed_shuffles[i,:num_shuffle] = Y_fed[i,:num_shuffle]
    X_full_shuffle = X_fed_shuffles.reshape(-1,d)
    Y_full_shuffle = Y_fed_shuffles.reshape(-1,1).squeeze()
    idx = np.random.permutation(X_full_shuffle.shape[0])
    X_full_shuffle[idx,:] = X_full_shuffle
    Y_full_shuffle[idx] = Y_full_shuffle 

    for i in range(clients):
        X_fed[i,:num_shuffle,:] = X_full_shuffle[i * num_shuffle:(i+1)*num_shuffle,:]
        Y_fed[i,:num_shuffle] = Y_full_shuffle[i*num_shuffle:(i+1)*num_shuffle]
    
    trialvals = []
    trialhets = []
    trialdeltas = []
    for t in range(trials):
        currtrial = []
        w = np.zeros((d,1))
        w_prev = np.zeros((d,1))
        GDA_flag = False
        eta_start = args.lr
        eta = eta_start
        eta_G = eta_start
        eta_E = 0
        s = 0
        rounds_passed = 0
        grad_norms = []
        aucs = []
        hets = []
        deltas = []
        for r in range(R):
            if eta_E > 0:
                beta = (1 - np.sqrt(mu * eta_E * K))/(1 + np.sqrt(mu * eta_E * K))
            else:
                beta = 0
            
            whalf = (1 + beta) * w - beta * w_prev

            w_grad_sum, oldgrads = grad_comp(n_0, 
                                        n_1, whalf, clients, 
                                    d, K, batch, X_fed, Y_fed, X_full, Y_full, mu,args, eta)
            w_prev = w
            w = whalf - (eta_G/float(clients)) * w_grad_sum
            w_grad_prev = w_grad_sum 
            if r % 1 == 0:
                w_grad_eval = wgrad(X_full, Y_full, w, mu)
                het_eval = het(w_grad_eval, X_fed, Y_fed, w, mu, clients)
                loss_eval = loss(X_full,Y_full,w,mu)
                grad_norm = np.linalg.norm(w_grad_eval)**2
                param_norm = np.linalg.norm(w)**2
                print('Trial {6} Stage {2}, eta {3}, eta_G {4} eta_E {7} Round {0}: Grad {1}  Loss {5} Het {8} Delta {9}'.format(r, 
                    grad_norm, s, eta, eta_G, loss_eval, t, eta_E, het_eval, param_norm))
                grad_norms.append(grad_norm)
                hets.append(het_eval)
                deltas.append(param_norm)
                currtrial.append(grad_norm)
        
            if args.multistage == 1:
                s, rounds_passed, eta, eta_G, GDA_flag, eta_E= local_schedule(eta_start, s, rounds_passed, args.stageone, R, K, GDA_flag, args.opt)
            if args.multistage == 2:
                eta_G, eta, eta_E = switch_schedule(eta_start, r, R, args.stageone, K, args.opt)
            if args.multistage == 3:
                s, rounds_passed, eta_G = simple_schedule(eta_start, s, rounds_passed, args.stageone, R)
                eta = eta_G
        trialvals.append(currtrial)
        trialdeltas.append(deltas)
        trialhets.append(hets)

    dirpath = os.path.join(dir_path, 'logistic/rounds{5}/batch{0}/FED_{2}_{4}_{6}/stage{1}/stageone{3}'.format(args.batch, args.multistage, args.control,args.stageone, args.opt, args.R, args.shuffle_pct))
    expname = 'lr{0}'.format(args.lr)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath,  exist_ok=True)
    
    
    gradvals, gradmeans, gradbars = make_bars(trialvals)
    hetvals, hetmeans, hetbars = make_bars(trialhets)
    deltavals, deltameans, deltabars = make_bars(trialdeltas)

    plt.errorbar(range(gradmeans.shape[0]), gradmeans, yerr = gradbars)
    plt.xlabel('Rounds')
    plt.ylabel('Grad Norm Squared')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(dir_path, os.path.join(dirpath, expname + '_grads' + '.png')))
    plt.clf()
    np.save(os.path.join(dir_path, os.path.join(dirpath, expname + '_grads')), gradvals)


    plt.errorbar(range(hetmeans.shape[0]), hetmeans, yerr = hetbars)
    plt.xlabel('Rounds')
    plt.ylabel('Heterogeneity')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(dir_path, os.path.join(dirpath, expname + '_hets' + '.png')))
    plt.clf()
    np.save(os.path.join(dir_path, os.path.join(dirpath, expname + '_hets')), hetvals)
    
    plt.errorbar(range(deltameans.shape[0]), deltameans, yerr = deltabars)
    plt.xlabel('Rounds')
    plt.ylabel('Delta')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(dir_path, os.path.join(dirpath, expname + '_deltas' + '.png')))
    plt.clf()
    np.save(os.path.join(dir_path, os.path.join(dirpath, expname + '_deltas')), deltavals)




def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--lr", type = float, default = 0.01)
    CLI.add_argument("--multistage", type = int, default = 0)
    CLI.add_argument("--control", type = int, default = 0)
    CLI.add_argument("--R", type = int, default = 100)
    CLI.add_argument("--stageone", type = float, default = 0.5)
    CLI.add_argument("--K", type = int, default = 20)
    CLI.add_argument("--batch", type = float, default = 1)
    CLI.add_argument("--GDA", type = int, default = 0)
    CLI.add_argument("--extra", type = int, default = 0)
    CLI.add_argument("--opt", type = int, default = 0)
    CLI.add_argument("--shuffle_pct", type=float, default = 1.)
    args = CLI.parse_args()
    print(args)
    if args.GDA == 1:
        run_GDA(args)
    elif args.extra == 1:
        run_extragrad(args)
    else:
        run_experiment(args)

if __name__ == "__main__":
    main()