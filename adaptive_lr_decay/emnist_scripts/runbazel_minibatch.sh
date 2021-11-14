cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/longexp_m_grid/$4EMNIST_minibatch_$1_switch_$2_multi_$3 run :federated_trainer -- --task=emnist_cr \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd \
--server_learning_rate=$1 --switch_round=$2 --swapped 1 --multistage=$3 --client_epochs_per_round=20 \
--experiment_name=$4EMNIST_Minibatch_$1_switch_$2_multi_$3