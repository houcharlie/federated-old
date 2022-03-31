# Copyright 2022, Google LLC.
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
"""Runs HypCluster on EMNIST with varying levels of data paucity."""

import collections
import functools
import math
from typing import Callable, List, Tuple

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff


from data_poor_fl import hypcluster
from data_poor_fl import personalization_utils
from data_poor_fl import pseudo_client_data
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as training_flags:
  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/data_poor_fl/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 100, 'Number of total training rounds.')

  # Train client configuration
  flags.DEFINE_integer('clients_per_train_round', 10,
                       'How many clients to sample at each training round.')
  flags.DEFINE_integer('examples_per_pseudo_client', 100,
                       'Maximum number of examples per pseudo-client.')
  flags.DEFINE_integer(
      'train_epochs', 1,
      'Number of epochs performed by a client during a round of training.')
  flags.DEFINE_integer('train_batch_size', 10, 'Batch size on train clients.')

  # Training algorithm configuration
  flags.DEFINE_bool('warmstart_hypcluster', False,
                    'Whether to warm-start HypCluster.')
  flags.DEFINE_string(
      'warmstart_root_dir', '',
      'Directory to load checkpoints from previous FedAvg training. Only used '
      'when `warmstart_hypcluster` is True.')
  flags.DEFINE_integer('num_clusters', 1,
                       'Number of clusters used in HypCluster.')

  # Eval algorithm configuration
  flags.DEFINE_integer(
      'clients_per_evaluation', 100, 'Number of clients sampled to perform '
      'federated evaluation.')

  # Random seeds for reproducibility
  flags.DEFINE_integer(
      'base_random_seed', 0, 'An integer random seed governing'
      ' the randomness in the simulation.')

  # Debugging flags
  flags.DEFINE_bool(
      'use_synthetic_data', False, 'Whether to use synthetic data. This should '
      'only be set to True for debugging purposes.')

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

FLAGS = flags.FLAGS

# Change constant to a flag if needs to be configured.
_ROUNDS_PER_EVALUATION = 10
_ROUNDS_PER_CHECKPOINT = 50
_EMNIST_MAX_ELEMENTS_PER_CLIENT = 418
# EMNIST has 3400 clients, we use the training data from 2500 clients for
# training, and the training data from the rest 900 clients for evaluation.
_NUM_RAW_EVAL_CLIENTS = 900


def _write_hparams():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(training_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Write the updated hyperparameters to a file.
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def _convert_keras_optimizer_to_tff(
    keras_optimizer: tf.keras.optimizers.Optimizer
) -> tff.learning.optimizers.Optimizer:
  """Creates a TFF optimizer equivalent of the given Keras optimizer."""
  if isinstance(keras_optimizer, tf.keras.optimizers.SGD):
    learning_rate = float(keras_optimizer.learning_rate.numpy())
    momentum = float(keras_optimizer.momentum.numpy())
    return tff.learning.optimizers.build_sgdm(
        learning_rate=learning_rate, momentum=momentum)
  elif isinstance(keras_optimizer, tf.keras.optimizers.Adam):
    learning_rate = float(keras_optimizer.learning_rate.numpy())
    beta_1 = float(keras_optimizer.beta_1.numpy())
    beta_2 = float(keras_optimizer.beta_2.numpy())
    epsilon = float(keras_optimizer.epsilon)
    return tff.learning.optimizers.build_adam(learning_rate, beta_1, beta_2,
                                              epsilon)
  else:
    raise TypeError(
        f'Expect a SGD or Adam Keras optimizers, found a {keras_optimizer}. '
        'Please update the function `_convert_keras_optimizer_to_tff` to '
        'support conversion from this Keras optimizer to a TFF optimizer.')


def _load_init_model_weights(
    model_fn: Callable[[],
                       tff.learning.Model]) -> List[tff.learning.ModelWeights]:
  """Load model weights to warm-start HypCluster."""
  state_manager = tff.program.FileProgramStateManager(FLAGS.warmstart_root_dir)
  learning_process_for_metedata = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      server_optimizer_fn=lambda: tf.keras.optimizers.Adam(1.0),
      client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES,
      model_aggregator=tff.learning.robust_aggregator(
          zeroing=True, clipping=True, add_debug_measurements=True))
  init_state = learning_process_for_metedata.initialize()
  loaded_models = []
  versions_saved = state_manager.versions()
  if FLAGS.num_clusters >= len(versions_saved):
    raise ValueError(
        f'The checkpoint directory {FLAGS.warmstart_root_dir} only has '
        f'{len(versions_saved)-1} checkpoints, but expected to load '
        f'{FLAGS.num_clusters} models. Please use a smaller value for '
        'FLAGS.num_clusters, or use a different checkpoint directory.')
  for i in range(1, FLAGS.num_clusters + 1):
    version = versions_saved[-i]
    state = state_manager.load(version=version, structure=init_state)
    loaded_models.append(learning_process_for_metedata.get_model_weights(state))
  return loaded_models


def _create_train_algorithm(
    model_fn: Callable[[], tff.learning.Model]
) -> tff.learning.templates.LearningProcess:
  """Creates a learning process for HypCluster training."""
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  # Need to set `no_nan_division=True` to avoid NaNs in the learned model, which
  # can happen when a model is not selected by any client in a round.
  model_aggregator = tff.aggregators.MeanFactory(no_nan_division=True)
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  # TODO(b/227488363): Delete `_convert_keras_optimizer_to_tff` once we have a
  # uitility function that directly creates a TFF optimizer from flags.
  client_tff_optimizer = _convert_keras_optimizer_to_tff(client_optimizer_fn())
  server_tff_optimizer = _convert_keras_optimizer_to_tff(server_optimizer_fn())
  initial_model_weights_list = None
  if FLAGS.warmstart_hypcluster:
    if not FLAGS.warmstart_root_dir:
      raise ValueError('Must provide a `warmstart_root_dir` when '
                       '`warmstart_hypcluster` is True.')
    initial_model_weights_list = _load_init_model_weights(model_fn)
  return hypcluster.build_hypcluster_train(
      model_fn=model_fn,
      num_clusters=FLAGS.num_clusters,
      client_optimizer=client_tff_optimizer,
      server_optimizer=server_tff_optimizer,
      model_aggregator=model_aggregator,
      initial_model_weights_list=initial_model_weights_list)


def _build_hypcluster_eval(model_fn: Callable[[], tff.learning.Model],
                           num_clusters: int,
                           client_data_type: tff.Type) -> tff.Computation:
  """Builds a computation for performing HypCluster evaluation.

  This function is similar to `hypcluster.build_hypcluster_evel`, except that:
  1. This function accepts an additional `client_data_type`, where we split the
     client-side input into two datasets: "selection_data" (used to select the
     best model) and "test_data" (used to evaluate the selected model).
  2. This function adds additional metrics such as: performance of individual
     models, and the percentage that each model is selected.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    num_clusters: An integer specifying the number of clusters to use.
    client_data_type: The `tff.Type` of client-side local input, which is an
      `OrderedDict` with two keys "selection_data" and "test_data".

  Returns:
    A federated TFF computation that performs HypCluster evaluation.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(model)
    unfinalized_metrics_type = tff.types.type_from_tensors(
        model.report_local_unfinalized_metrics())
    metrics_aggregation_fn = tff.learning.metrics.sum_then_finalize(
        model.metric_finalizers(), unfinalized_metrics_type)

  metrics_gather_fn = hypcluster.build_gather_fn(unfinalized_metrics_type,
                                                 num_clusters)
  list_weights_type = tff.StructWithPythonType(
      [model_weights_type for _ in range(num_clusters)], container_type=list)

  @tff.tf_computation(list_weights_type, client_data_type)
  def local_hypcluster_eval(model_weights, dataset):
    eval_models = [model_fn() for _ in range(num_clusters)]
    eval_models_outputs_for_select = hypcluster.multi_model_eval(
        eval_models, model_weights, dataset['selection_data'])
    eval_models_outputs_for_metrics = hypcluster.multi_model_eval(
        eval_models, model_weights, dataset['test_data'])
    best_model_index = hypcluster.select_best_model(
        eval_models_outputs_for_select)
    local_metrics = collections.OrderedDict(
        best=metrics_gather_fn(eval_models_outputs_for_metrics,
                               best_model_index))
    for i in range(num_clusters):
      local_metrics[f'model_{i}'] = metrics_gather_fn(
          eval_models_outputs_for_metrics, i)
    for i in range(num_clusters):
      local_metrics[f'choose_{i}'] = tf.cast(
          tf.equal(best_model_index, i), tf.float32)
    return local_metrics

  @tff.federated_computation(
      tff.type_at_server(list_weights_type),
      tff.type_at_clients(client_data_type))
  def hypcluster_eval(server_model_weights, client_datasets):
    client_model_weights = tff.federated_broadcast(server_model_weights)
    client_metrics = tff.federated_map(local_hypcluster_eval,
                                       (client_model_weights, client_datasets))
    eval_metrics = collections.OrderedDict()
    metric_names = tff.structure.name_list(
        local_hypcluster_eval.type_signature.result)
    for name in metric_names:
      if 'choose' in name:
        eval_metrics[name] = tff.federated_mean(client_metrics[name])
      else:
        eval_metrics[name] = metrics_aggregation_fn(client_metrics[name])
    return tff.federated_zip(eval_metrics)

  return hypcluster_eval


def _get_pseudo_client_ids(examples_per_pseudo_clients: int,
                           base_client_examples_df: pd.DataFrame,
                           separator: str = '-') -> List[str]:
  """Generates a list of pseudo-client ids."""
  pseudo_client_ids = []
  for _, row in base_client_examples_df.iterrows():
    num_pseudo_clients = math.ceil(row.num_examples /
                                   examples_per_pseudo_clients)
    client_id = row.client_id
    expanded_client_ids = [
        client_id + separator + str(i) for i in range(num_pseudo_clients)
    ]
    pseudo_client_ids += expanded_client_ids
  return pseudo_client_ids


def _split_pseudo_client_ids(
    raw_client_ids: List[str],
    pseudo_client_ids: List[str],
    separator: str = '-') -> Tuple[List[str], List[str]]:
  """Splits the pseudo-client ids into training and evaluation."""
  random_state = np.random.RandomState(seed=FLAGS.base_random_seed)
  shuffled_raw_client_ids = random_state.permutation(raw_client_ids)
  raw_eval_client_ids = shuffled_raw_client_ids[:_NUM_RAW_EVAL_CLIENTS]
  pseudo_train_client_ids = []
  pseudo_eval_client_ids = []
  for pseudo_client_id in pseudo_client_ids:
    raw_id, _ = pseudo_client_id.split(separator)
    if raw_id in raw_eval_client_ids:
      pseudo_eval_client_ids.append(pseudo_client_id)
    else:
      pseudo_train_client_ids.append(pseudo_client_id)
  return pseudo_train_client_ids, pseudo_eval_client_ids


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  if not FLAGS.experiment_name:
    raise ValueError('FLAGS.experiment_name must be set.')

  # Configuring the base EMNIST task
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.train_batch_size,
      shuffle_buffer_size=_EMNIST_MAX_ELEMENTS_PER_CLIENT)
  task = tff.simulation.baselines.emnist.create_character_recognition_task(
      train_client_spec,
      model_id='cnn',
      only_digits=False,
      use_synthetic_data=FLAGS.use_synthetic_data)
  train_preprocess_fn = task.datasets.train_preprocess_fn
  eval_preprocess_fn = task.datasets.eval_preprocess_fn

  # Creating pseudo-clients
  if not FLAGS.use_synthetic_data:
    csv_file_path = 'data_poor_fl/emnist_train_num_examples.csv'
    with open(csv_file_path) as csv_file:
      train_client_example_counts = pd.read_csv(csv_file)
    separator = '-'
    pseudo_client_ids = _get_pseudo_client_ids(FLAGS.examples_per_pseudo_client,
                                               train_client_example_counts,
                                               separator)
    pseudo_train_client_ids, pseudo_eval_client_ids = _split_pseudo_client_ids(
        task.datasets.train_data.client_ids, pseudo_client_ids, separator)
  else:
    pseudo_train_client_ids, pseudo_eval_client_ids = None, None

  extended_train_data = pseudo_client_data.create_pseudo_client_data(
      base_client_data=task.datasets.train_data,
      examples_per_pseudo_client=FLAGS.examples_per_pseudo_client,
      pseudo_client_ids=pseudo_train_client_ids)
  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          extended_train_data.client_ids, random_seed=FLAGS.base_random_seed),
      size=FLAGS.clients_per_train_round)
  extended_eval_data = pseudo_client_data.create_pseudo_client_data(
      base_client_data=task.datasets.train_data,
      examples_per_pseudo_client=FLAGS.examples_per_pseudo_client,
      pseudo_client_ids=pseudo_eval_client_ids)

  # Creating the training process (and wiring in a dataset computation)
  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = extended_train_data.dataset_computation(client_id)
    return train_preprocess_fn(raw_client_data)

  learning_process = _create_train_algorithm(task.model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, learning_process)
  training_process.get_model_weights = learning_process.get_model_weights

  @tff.tf_computation(tf.string)
  def build_eval_datasets_from_client_id(client_id):
    raw_data = extended_eval_data.dataset_computation(client_id)
    # Unbatching before splitting the data into half. This allows splitting at
    # the example level instead of at the batch level.
    reshaped_data = eval_preprocess_fn(raw_data).unbatch()
    selection_data, test_data = personalization_utils.split_half(reshaped_data)
    return collections.OrderedDict(
        selection_data=selection_data.batch(FLAGS.train_batch_size),
        test_data=test_data.batch(FLAGS.train_batch_size))

  hypcluster_eval = _build_hypcluster_eval(
      model_fn=task.model_fn,
      num_clusters=FLAGS.num_clusters,
      client_data_type=build_eval_datasets_from_client_id.type_signature.result)
  # Compose the dataset computation with the hypcluster eval computation. Note
  # that `tff.simulation.compose_dataset_computation_with_computation` does not
  # work when the dataset computation returns a dict of two datasets.
  model_weights_at_clients_type = tff.types.at_server(
      training_process.get_model_weights.type_signature.result)

  @tff.federated_computation(model_weights_at_clients_type,
                             tff.types.at_clients(tf.string))
  def composed_dataset_comp_with_hypcluster_eval(model_weights, client_ids):
    processed_datasets = tff.federated_map(build_eval_datasets_from_client_id,
                                           client_ids)
    return hypcluster_eval(model_weights, processed_datasets)

  def evaluation_fn(state, federated_data):
    return composed_dataset_comp_with_hypcluster_eval(
        training_process.get_model_weights(state), federated_data)

  evaluation_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          extended_eval_data.client_ids, random_seed=FLAGS.base_random_seed),
      size=FLAGS.clients_per_evaluation)

  # Configuring release managers and performing training/eval
  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparams()
  tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=_ROUNDS_PER_EVALUATION,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=_ROUNDS_PER_CHECKPOINT,
      metrics_managers=metrics_managers)


if __name__ == '__main__':
  app.run(main)
