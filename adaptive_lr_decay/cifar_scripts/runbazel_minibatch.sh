cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/cifarcore/CIFAR_minibatch_$1_switch_$2_multi_$3 run :federated_trainer -- --task=cifar100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --swapped 1 --multistage=$3 \
--experiment_name=CIFAR_Minibatch_$1_switch_$2_multi_$3