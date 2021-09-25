cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/EMNIST_$1_switch_$2_control_$3 run :federated_trainer -- --task=emnist_cr --total_rounds=100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --clients_per_round=100 --switch_round=$2 --control=$3 \
--experiment_name=EMNIST_$1_switch_$2_control_$3