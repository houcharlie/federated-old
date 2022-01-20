# Adaptive Learning Rate Decay

This directory contains libraries for performing federated learning with
adaptive learning rate decay. For a more general look at using TensorFlow
Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).
This directory contains a more advanced version of federated averaging, and
assumes some familiarity with libraries such as
[tff.learning.build_federated_averaging_process](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_federated_averaging_process).

## Dependencies

To use this library, one should first follow the instructions
[here](https://github.com/tensorflow/federated/blob/master/docs/install.md) to
install TensorFlow Federated using pip. Other pip packages are required by this
library, and may need to be installed. They can be installed via the following
commands:

```
pip install absl-py
pip install attr
pip install dm-tree
pip install numpy
pip install pandas
pip install tensorflow
```

## General description

This example contains two main libraries,
[adaptive_fed_avg.py](https://github.com/google-research/federated/blob/master/adaptive_lr_decay/adaptive_fed_avg.py)
and
[callbacks.py](https://github.com/google-research/federated/blob/master/adaptive_lr_decay/callbacks.py).
The latter implements learning rate callbacks that adaptively decay learning
rates based on moving averages of metrics. This is relevant in the federated
setting, as we may wish to decay learning rates based on the average training
loss across rounds.

These callbacks are used in `adaptive_fed_avg.py` to perform federated averaging
with adaptive learning rate decay. Notably, `adaptive_fed_avg.py` decouples
client and server leaerning rates so that they can be decayed independently, and
so that we do not conflate their effects. In order to do this adaptive decay,
the iterative process computes metrics before and during training. The metrics
computed before training are used to inform the learning rate decay throughout.

## Example usage

Suppose we wanted to run a training process where we switch the algorithm from a local
update algorithm to a global update algorithm after 500 rounds.  Then we would make
a client/server learning rate callback as

```
client_lr_callback = callbacks.create_switch_lr(
        owner='Client',
        learning_rate=0.1,
        start_lr=0.1,
        decay_factor=1.,
        switch_round=500,
        swapped=false)
server_lr_callback = callbacks.create_switch_lr(
        owner='Server',
        learning_rate=0.1,
        start_lr=0.1,
        decay_factor=1.,
        switch_round=500,
        swapped=false)
```

These callbacks are incorporated into `adaptive_fed_avg` so that the algorithm
will know exactly when to switch from, say, FedAvg, to SGD.

## More detailed usage

There are other options in federated_trainer.py that allow one to vary how
exactly one controls the server and client learning rates to transition from
a local update method to a global update method like SGD.  Most notable are:

multistage: a flag that says whether we are using learning rate decay in the 
fedavg/sgd components of the algorithm.

switch_round: if multistage==0, then it denotes the proportion of rounds before
switching to SGD.  if multistage==1, then it denotes the proportion of rounds before
the first learning rate decay event.

swap_round: if multistage==1, it denotes the proportion of rounds before switching
to SGD.

## Instructions on running experiments

To run the corresponding binaries, we require [Bazel](https://www.bazel.build/).
Instructions for installing Bazel can be found
[here](https://docs.bazel.build/versions/master/install.html).

To run federated chaining (without learning rate decay, 
and switching to SGD after 50% of rounds) on CIFAR-100, for example, 
one would run (inside this directory):

```
bazel run :federated_trainer -- --task=cifar100 \
--client_optimizer=sgd --client_learning_rate=0.1 --server_optimizer=sgd \
--server_learning_rate=0.1 --switch_round=0.5 --total_rounds=100 --clients_per_round=10 --client_epochs_per_round=1 \
--experiment_name=cifar100_classification
```

This will run 50 communication rounds of federated averaging, using SGD on both
the server and client, then 50 communication rounds of pure SGD,
 with 10 clients per round and 1 client epoch per round.
For more details on these flags, see
[federated_trainer.py](https://github.com/google-research/federated/blob/master/adaptive_lr_decay/federated_trainer.py).
