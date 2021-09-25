bazel run :federated_trainer -- --task=emnist_cr --total_rounds=100 \
--client_optimizer=sgd --client_learning_rate=0.1 --server_optimizer=sgd \
--server_learning_rate=0.1 --clients_per_round=10 --client_epochs_per_round=1 \
--experiment_name=emnist_classification_test_v2
