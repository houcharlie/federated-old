cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel run :federated_trainer -- --task=emnist_cr --total_rounds=10 \
--client_optimizer=sgd --client_learning_rate=0.1 --server_optimizer=sgd \
--server_learning_rate=0.1 --clients_per_round=10 --switch_round=0.2 --control=1 --multistage=1 \
--experiment_name=emnist_TEST10