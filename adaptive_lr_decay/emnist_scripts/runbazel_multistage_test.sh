cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/2epoch/EMNISTMULTI_$1_switch_$2_control_$3_allowswap_$4 run :federated_trainer -- --task=emnist_cr \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --control=$3 --allow_swap=$4 --multistage=1 --rounds_per_eval=100 --root_output_dir=/ocean/projects/iri180031p/houc/multistage/low_epochs_test \
--experiment_name=EMNISTMULTI_$1_switch_$2_control_$3_allowswap_$4