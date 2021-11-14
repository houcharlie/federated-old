cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel --output_base=/tmp/houc/longexp_m_grid/cifar_$1_minibatch_$2 run :federated_trainer -- --task=cifar100 \
--client_optimizer=sgd --client_learning_rate=$1 --server_optimizer=sgd --swap_round=200 \
--server_learning_rate=$1 --switch_round=0.2 --swapped=$2 --multistage=1 --client_epochs_per_round=1 --total_rounds=5000 --rounds_per_eval=50 \
--experiment_name=CIFAR_generalize_$1_minibatch_$2