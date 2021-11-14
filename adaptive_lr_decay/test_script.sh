cd /jet/home/houc/multistage/federated/adaptive_lr_decay
bazel run :federated_trainer -- --task=emnist_cr \
--client_optimizer=sgd --client_learning_rate=0.1 --server_optimizer=sgd \
--server_learning_rate=0.1 --switch_round=0.7 --control=0 --multistage=0 --total_rounds=10 --rounds_per_eval=1 \
--experiment_name=test_switch2