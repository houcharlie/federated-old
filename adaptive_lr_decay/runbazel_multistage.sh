cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel run :federated_trainer -- --task=emnist_cr --total_rounds=100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --clients_per_round=10 --switch_round=$2 --control=$3 --allow_swap=$4 --multistage=1 \
--experiment_name=EMNISTMULTI_sample_10_lr_$1_switch_$2_control_$3_allowswap_$4