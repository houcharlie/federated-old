cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/orthogonality/EMNIST_$1_switch_$2_control_$3 run :federated_trainer -- --task=emnist_cr \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd --client_epochs_per_round=20 --total_rounds=500 \
--server_learning_rate=$1 --switch_round=$2 --control=$3 --root_output_dir=/ocean/projects/iri180031p/houc/multistage/orthogonality \
--experiment_name=EMNIST_$1_switch_$2_control_$3