cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/longexp_m_grid/EMNIST_$1_minibatch_$2 run :federated_trainer -- --task=emnist_cr \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd --swap_round=200 \
--server_learning_rate=$1 --switch_round=0.2 --swapped=$2 --multistage=1 --client_epochs_per_round=1 --total_rounds=10000 --rounds_per_eval=50 \
--experiment_name=EMNIST_generalize_$1_minibatch_$2_long