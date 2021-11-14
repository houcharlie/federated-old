import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os

#mbsgd_file = '/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/EMNIST_generalize_0.4_minibatch_1/experiment.metrics.csv'
mbsgd_file = '/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFAR_generalize_0.4_minibatch_1/experiment.metrics.csv'
df = pd.read_csv(mbsgd_file, sep=',')
y = df['train_eval/loss'][::50]
x = range(0, len(y)*50, 50)
plt.plot(x, y, label='Minibatch SGD')

#fedavg_file = '/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/EMNIST_generalize_0.2_minibatch_0/experiment.metrics.csv'
fedavg_file = '/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp/results/CIFAR_generalize_0.3_minibatch_0/experiment.metrics.csv'

df = pd.read_csv(fedavg_file, sep=',')
y = df['train_eval/loss'][::50]
x = range(0, len(y)*50, 50)
plt.plot(x, y, label='Local SGD')
plt.legend()
plt.savefig('/jet/home/houc/multistage/federated/adaptive_lr_decay/emnist_train_plot.png')