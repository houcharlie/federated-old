cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/CIFARcore/$5CIFARMULTI_rerun_$1_switch_$2_control_$3_allowswap_$4 run :federated_trainer -- --task=cifar100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --control=$3 --swap_round=$4 --multistage=1 --total_rounds=5000 --rounds_per_eval=200 --rounds_per_checkpoint=200 \
--experiment_name=CIFARMULTILONGREDO_3_$1_switch_$2_control_$3_swapround_$4