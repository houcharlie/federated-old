cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/CIFARcore/CIFARMULTI_$1_switch_$2_control_$3_allowswap_$4 run :federated_trainer -- --task=cifar100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --control=$3 --allow_swap=$4 --multistage=1 \
--experiment_name=CIFARMULTI_$1_switch_$2_control_$3_allowswap_$4