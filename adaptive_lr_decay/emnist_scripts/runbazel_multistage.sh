cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/longexp_m_grid/$5EMNISTMULTI_$1_switch_$2_control_$3_allowswap_$4 run :federated_trainer -- --task=emnist_cr \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --control=$3 --swap_round=$4 --multistage=1 --client_epochs_per_round=20 \
--experiment_name=$5EMNISTMULTI_3_$1_switch_$2_control_$3_swapround_$4