cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/cifarcore/$4CIFAR_$1_switch_$2_control_$3 run :federated_trainer -- --task=cifar100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --control=$3 --total_rounds=5000\
--experiment_name=CIFARLONG_$1_switch_$2_control_$3