# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated CIFAR-100 classification library using TFF."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import training_specs
from utils import training_utils
from utils.datasets import cifar100_dataset
from utils.models import resnet_models

CIFAR_SHAPE = (32, 32, 3)
NUM_CLASSES = 100


def configure_training(
    task_spec: training_specs.TaskSpec,
    crop_size: int = 24,
    distort_train_images: bool = True,
    cache_dir: str = '~') -> training_specs.RunnerSpec:
  """Configures training for the CIFAR-100 classification task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    crop_size: An optional integer representing the resulting size of input
      images after preprocessing.
    distort_train_images: A boolean indicating whether to distort training
      images during preprocessing via random crops, as opposed to simply
      resizing the image.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """
  crop_shape = (crop_size, crop_size, 3)

  cifar_train, _ = tff.simulation.datasets.cifar100.load_data(cache_dir=cache_dir)
  _, cifar_test = cifar100_dataset.get_centralized_datasets(
      train_batch_size=task_spec.client_batch_size, crop_shape=crop_shape, cache_dir=cache_dir)

  train_preprocess_fn = cifar100_dataset.create_preprocess_fn(
      num_epochs=task_spec.client_epochs_per_round,
      batch_size=task_spec.client_batch_size,
      crop_shape=crop_shape,
      distort_image=distort_train_images)
  input_spec = train_preprocess_fn.type_signature.result.element

  model_builder = functools.partial(
      resnet_models.create_resnet18,
      input_shape=crop_shape,
      num_classes=NUM_CLASSES)

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(tff_model_fn)

  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    client_dataset = cifar_train.dataset_computation(client_id)
    return train_preprocess_fn(client_dataset)

  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, iterative_process)
  client_ids_fn = training_utils.build_sample_fn(
      cifar_train.client_ids,
      size=task_spec.clients_per_round,
      replace=False,
      random_seed=task_spec.client_datasets_random_seed)
  # We convert the output to a list (instead of an np.ndarray) so that it can
  # be used as input to the iterative process.
  client_sampling_fn = lambda x: list(client_ids_fn(x))

  training_process.get_model_weights = iterative_process.get_model_weights

  test_fn = training_utils.build_centralized_evaluate_fn(
      eval_dataset=cifar_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  validation_fn = lambda model_weights, round_num: test_fn(model_weights)

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
