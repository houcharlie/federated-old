load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//google/catalyst/main:__subpackages__"])

licenses(["notice"])

py_binary(
    name = "centralized_trainer",
    srcs = ["centralized_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//catalyst/cifar100:centralized_cifar100",
        "//catalyst/emnist:centralized_emnist",
        "//catalyst/emnist_ae:centralized_emnist_ae",
        "//catalyst/shakespeare:centralized_shakespeare",
        "//catalyst/shared:optimizer_utils",
        "//catalyst/stackoverflow:centralized_stackoverflow",
        "//catalyst/stackoverflow_lr:centralized_stackoverflow_lr",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "centralized_tasks_test",
    size = "large",
    srcs = ["centralized_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [
        "//catalyst/cifar100:centralized_cifar100",
        "//catalyst/emnist:centralized_emnist",
        "//catalyst/emnist_ae:centralized_emnist_ae",
        "//catalyst/shakespeare:centralized_shakespeare",
        "//catalyst/stackoverflow:centralized_stackoverflow",
        "//catalyst/stackoverflow_lr:centralized_stackoverflow_lr",
    ],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//catalyst/cifar100:federated_cifar100",
        "//catalyst/emnist:federated_emnist",
        "//catalyst/emnist_ae:federated_emnist_ae",
        "//catalyst/shakespeare:federated_shakespeare",
        "//catalyst/shared:fed_avg_schedule",
        "//catalyst/shared:optimizer_utils",
        "//catalyst/shared:training_specs",
        "//catalyst/stackoverflow:federated_stackoverflow",
        "//catalyst/stackoverflow_lr:federated_stackoverflow_lr",
        "//utils:training_loop",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "federated_tasks_test",
    size = "large",
    srcs = ["federated_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [
        "//catalyst/cifar100:federated_cifar100",
        "//catalyst/emnist:federated_emnist",
        "//catalyst/emnist_ae:federated_emnist_ae",
        "//catalyst/shakespeare:federated_shakespeare",
        "//catalyst/shared:training_specs",
        "//catalyst/stackoverflow:federated_stackoverflow",
        "//catalyst/stackoverflow_lr:federated_stackoverflow_lr",
        "//utils:training_loop",
    ],
)
