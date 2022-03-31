# Copyright 2021, Google LLC.
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
"""Runs federated training with large cohorts on a number of tasks."""

import collections
import functools
import os.path
from typing import Callable

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import aggregation
from large_cohort import data_utils
from large_cohort import file_utils
from large_cohort import flag_utils
from large_cohort import warmup
from utils import task_utils
from utils import training_utils

with flag_utils.record_hparam_flags() as shared_flags:
  # Client configuration
  flags.DEFINE_integer(
      'train_epochs', 1,
      'Number of epochs each client performs during a single '
      'round of training.')
  flags.DEFINE_integer('train_batch_size', 32, 'Batch size on train clients.')
  flags.DEFINE_integer('eval_batch_size', 64, 'Batch size for evaluation.')
  flags.DEFINE_integer(
      'max_train_elements', None, 'Max number of examples used by each client '
      'when training. If set to `None`, all examples are used. This should only'
      ' be set to a positive value for Stack Overflow tasks.')
  flags.DEFINE_integer(
      'max_eval_elements', None, 'Max number of examples used by each client '
      'when evaluating. If set to `None`, all examples are used. This should '
      'only be set to a positive value for Stack Overflow tasks.')

  # Iterative process configuration
  flags.DEFINE_enum('iterative_process', 'FedOpt', ['FedOpt', 'FedSGD'],
                    'Which iterative process to use.')
  flags.DEFINE_bool(
      'uniform_weighting', False, 'Whether to use uniform '
      'weighting when aggregating client updates (True) or use '
      'example-weighted aggregation (False).')
  flags.DEFINE_bool('clipping', True, 'Whether to use adaptive clipping.')
  flags.DEFINE_bool('zeroing', True, 'Whether to use adaptive zeroing.')

  # Metrics, checkpoint, and training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/large_cohort/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer('total_rounds', 10, 'Number of total training rounds.')

  # Training configuration
  flags.DEFINE_integer('clients_per_train_round', 10,
                       'How many clients to sample at each training round.')
  flags.DEFINE_integer(
      'rounds_to_double_cohort', None,
      'The number of rounds that must elapsed before doubling the cohort size. '
      'If `None`, cohort size is fixed throughout training. Otherwise, must be '
      'a positive integer. Doubles the cohort size every N rounds, starting with '
      'a cohort size set in --clients_per_round')

  # Validation configuration
  flags.DEFINE_integer(
      'clients_per_validation_round', -1,
      'How many clients to sample at each validation round. If'
      'set to a positive number, we perform a federated '
      'evaluation periodically every `FLAGS.rounds_per_evaluation` round. '
      'Otherwise, we perform a centralized evaluation periodically.')
  flags.DEFINE_integer(
      'rounds_per_validation', 10,
      'How often to evaluate the model on a sample of validation clients.')
  flags.DEFINE_bool(
      'set_validation_to_test', True, 'Whether to use the test set to perform '
      'validation. This should be set to `False` only if an experiment is '
      'performing explicit validation.')
  flags.DEFINE_bool(
      'final_validation', False, 'Whether to perform a final validation round '
      'after training has completed.')

  # Random seeds for reproducibility
  flags.DEFINE_integer(
      'base_random_seed', 0, 'An integer random seed governing'
      ' the randomness in the simulation.')

with flag_utils.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

with flag_utils.record_hparam_flags() as optimizer_flags:
  flag_utils.define_optimizer_flags('client')
  flag_utils.define_optimizer_flags('server')
  flags.DEFINE_bool(
      'use_server_warmup', False, 'A boolean indicating whether '
      'the server should do learning rate warmup. If set to True,'
      'the server will perform a linear warmup scaling for the '
      'first 100 communication rounds.')

FLAGS = flags.FLAGS


def _validate_flags():
  """Validates that flag specifications are compatible."""
  if FLAGS.iterative_process == 'FedSGD' and FLAGS.client_optimizer is not None:
    raise ValueError('Found flag --client_optimizer_fn={!s}, but '
                     '--iterative_process=FedSGD. If using a client optimizer, '
                     'please set --iterative_process=FedOpt'.format(
                         FLAGS.client_optimizer))


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = flag_utils.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = flag_utils.lookup_flag_values(optimizer_flags)
  if FLAGS.iterative_process == 'FedOpt':
    opt_flag_dict = flag_utils.remove_unused_optimizer_flags(
        'client', opt_flag_dict)
  opt_flag_dict = flag_utils.remove_unused_optimizer_flags(
      'server', opt_flag_dict)
  hparam_dict = flag_utils.lookup_flag_values(task_flags)
  hparam_dict.update(opt_flag_dict)

  # Write the updated hyperparameters to a file.
  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  file_utils.create_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  file_utils.atomic_write_series_to_csv(hparam_dict, hparam_file)


def _create_iterative_process(
    model_fn: Callable[[],
                       tff.learning.Model]) -> tff.templates.IterativeProcess:
  """Creates an iterative process from a given model function."""
  server_optimizer_fn = flag_utils.create_optimizer_fn_from_flags('server')
  if FLAGS.use_server_warmup:
    warmup_schedule = warmup.WarmupSchedule(
        max_learning_rate=FLAGS.server_learning_rate, warmup_steps=100)
    server_optimizer_fn = functools.partial(
        server_optimizer_fn, learning_rate=warmup_schedule)

  model_update_aggregation_factory = aggregation.create_aggregator(
      zeroing=FLAGS.zeroing, clipping=FLAGS.clipping)

  if FLAGS.uniform_weighting:
    client_weighting = tff.learning.ClientWeighting.UNIFORM
  else:
    client_weighting = tff.learning.ClientWeighting.NUM_EXAMPLES

  if FLAGS.iterative_process == 'FedOpt':

    client_optimizer_fn = flag_utils.create_optimizer_fn_from_flags('client')
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weighting,
        model_aggregator=model_update_aggregation_factory)

  elif FLAGS.iterative_process == 'FedSGD':

    return tff.learning.algorithms.build_fed_sgd(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        model_aggregator=model_update_aggregation_factory)  # pytype: disable=bad-return-type  # gen-stub-imports


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  _validate_flags()

  if FLAGS.rounds_to_double_cohort is None:
    rounds_to_double_cohort = 0
  else:
    rounds_to_double_cohort = FLAGS.rounds_to_double_cohort
  train_sampling_seed = int('{}{}{}'.format(FLAGS.clients_per_train_round,
                                            FLAGS.base_random_seed,
                                            rounds_to_double_cohort))

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.train_batch_size,
      max_elements=FLAGS.max_train_elements)
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1,
      batch_size=FLAGS.eval_batch_size,
      max_elements=FLAGS.max_eval_elements)
  task = task_utils.create_task_from_flags(
      train_client_spec=train_client_spec, eval_client_spec=eval_client_spec)
  test_data = task.datasets.test_data

  # Split training data into train/validation data, if necessary.
  if FLAGS.set_validation_to_test:
    train_data = task.datasets.train_data
    validation_data = test_data
  elif task.datasets.validation_data is None:
    train_data, validation_data = data_utils.create_train_validation_split(
        task.datasets.train_data, seed=FLAGS.base_random_seed)
  else:
    train_data = task.datasets.train_data
    validation_data = task.datasets.validation_data

  # Create training artifacts
  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = train_data.dataset_computation(client_id)
    return task.datasets.train_preprocess_fn(raw_client_data)

  iterative_process = _create_iterative_process(task.model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, iterative_process)

  training_selection_fn = data_utils.create_sampling_fn(
      seed=train_sampling_seed,
      client_ids=train_data.client_ids,
      clients_per_round=FLAGS.clients_per_train_round,
      rounds_to_double_cohort=FLAGS.rounds_to_double_cohort)

  # Create evaluation aritfacts
  federated_evaluation_fn = tff.learning.build_federated_evaluation(
      task.model_fn)
  if FLAGS.clients_per_validation_round > 0:

    @tff.tf_computation(tf.string)
    def build_validation_dataset_from_client_id(client_id):
      raw_client_data = validation_data.dataset_computation(client_id)
      return task.datasets.eval_preprocess_fn(raw_client_data)

    validation_fn = tff.simulation.compose_dataset_computation_with_computation(
        build_validation_dataset_from_client_id, federated_evaluation_fn)

    evaluation_selection_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            sample_range=validation_data.client_ids,
            replace=False,
            random_seed=FLAGS.base_random_seed),
        size=FLAGS.clients_per_validation_round)

    def evaluation_fn(state, client_ids):  # pytype: disable=attribute-error  # gen-stub-imports
      return validation_fn(state.model, client_ids)

  else:

    full_validation_dataset = task.datasets.eval_preprocess_fn(
        validation_data.create_tf_dataset_from_all_clients())
    evaluation_selection_fn = lambda round_num: [full_validation_dataset]

    def evaluation_fn(state, evaluation_data):
      return federated_evaluation_fn(state.model, evaluation_data)

  # Configure and run the training loop
  _write_hparam_flags()
  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)

  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_validation,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  # Perform post-training evaluation
  full_train_dataset = task.datasets.eval_preprocess_fn(
      train_data.create_tf_dataset_from_all_clients())
  full_test_dataset = task.datasets.eval_preprocess_fn(
      test_data.create_tf_dataset_from_all_clients())

  post_training_metrics = collections.OrderedDict()

  post_training_metrics['train'] = federated_evaluation_fn(
      state.model, [full_train_dataset])
  post_training_metrics['test'] = federated_evaluation_fn(
      state.model, [full_test_dataset])

  if FLAGS.final_validation:
    full_validation_dataset = task.datasets.eval_preprocess_fn(
        validation_data.create_tf_dataset_from_all_clients())
    post_training_metrics['validation'] = federated_evaluation_fn(
        state.model, [full_validation_dataset])

  for metrics_manager in metrics_managers:
    metrics_manager.release(post_training_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
