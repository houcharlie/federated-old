# Copyright 2020, Google LLC.
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
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `shared/fed_avg_schedule.py`.
"""
import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow_federated as tff

from shrink_unshrink import models
from shrink_unshrink import shrink_unshrink_tff
from shrink_unshrink import simple_fedavg_tf
from shrink_unshrink import simple_fedavg_tff
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer(
      'max_elements_per_client', None, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt2/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 100,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer('small_embedding_size', 72,
                       'The embedding size of the lstm.')
  flags.DEFINE_integer('small_lstm_size', 503, 'The size of the lstm layer.')
  flags.DEFINE_float('oja_hyperparameter', 1.0,
                     'The hyperparameter used in learned_layerwise.')

  flags.DEFINE_integer('small_conv1_filters', 24,
                       'The filter size of the conv.')
  flags.DEFINE_integer('small_conv2_filters', 48,
                       'The filter size of the conv.')
  flags.DEFINE_integer('small_dense_size', 96, 'The size of the dense layer.')

  flags.DEFINE_integer('big_conv1_filters', 32, 'The filter size of the conv.')
  flags.DEFINE_integer('big_conv2_filters', 64, 'The filter size of the conv.')
  flags.DEFINE_integer('big_dense_size', 128, 'The size of the dense layer.')

  flags.DEFINE_integer(
      'new_projection_dict_decimate', 1,
      'The number of iterations before a new set of projection matrices are created.'
  )

  flags.DEFINE_integer(
      'static_client_layerwise_num_buckets', 1,
      'The number of hash buckets associated with static_client_layerwise_num_buckets.'
  )

  flags.DEFINE_enum(
      name='shrink_unshrink_type',
      default='identity',
      enum_values=[
          'identity', 'layerwise', 'client_layerwise', 'learned_layerwise',
          'learned_layerwise_v2', 'learned_sparse_layerwise_v2',
          'static_client_layerwise'
      ],
      help='what type of shrink_unshrink to do')

  flags.DEFINE_enum(  # originally only used to be cnn_dropout
      name='my_emnist_model_id',
      default='cnn_dropout',
      enum_values=['cnn_dropout', 'cnn', 'cnn_dropout_mfactor'],
      help='what type of cnn to use')

  flags.DEFINE_enum(
      name='build_projection_matrix_type',
      default='normal',
      enum_values=['normal', 'dropout', 'qr', 'sparse', 'ensemble'],
      help='what type of shrink_unshrink to do')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task flags
  task_flag_dict = utils_impl.lookup_flag_values(task_flags)
  hparam_dict.update(task_flag_dict)
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  logging.info('beginning main')
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  if FLAGS.task == 'stackoverflow_word':
    big_model_fn, small_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        task,
        big_embedding_size=96,
        big_lstm_size=670,
        small_embedding_size=FLAGS.small_embedding_size,
        small_lstm_size=FLAGS.small_lstm_size)
    # allows for modifications to lstm layers
    left_mask = [-1, 0, 2, -1, 2, -1, 0, -1]
    right_mask = [0, 1, 1, 1, 0, 0, -1, -1]

    # does not allow for modifications to lstm layers
    # left_mask = [-1, 0, -1, -1, 2, -1, 0, -1]
    # right_mask = [0, -1, -1, -1, 0, 0, -1, -1]
  elif FLAGS.task == 'emnist_character':
    if FLAGS.my_emnist_model_id == 'cnn_dropout_mfactor':
      # originally only used to be cnn_dropout
      big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_dropout_mfactor_model_fn(
          task,
          big_conv1_filters=FLAGS.big_conv1_filters,
          big_conv2_filters=FLAGS.big_conv2_filters,
          big_dense_size=FLAGS.big_dense_size,
          small_conv1_filters=FLAGS.small_conv1_filters,
          small_conv2_filters=FLAGS.small_conv2_filters,
          small_dense_size=FLAGS.small_dense_size)
    elif FLAGS.my_emnist_model_id == 'cnn_dropout':
      big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_dropout_model_fn(
          task,
          big_conv1_filters=FLAGS.big_conv1_filters,
          big_conv2_filters=FLAGS.big_conv2_filters,
          big_dense_size=FLAGS.big_dense_size,
          small_conv1_filters=FLAGS.small_conv1_filters,
          small_conv2_filters=FLAGS.small_conv2_filters,
          small_dense_size=FLAGS.small_dense_size)
    elif FLAGS.my_emnist_model_id == 'cnn':
      big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_model_fn(
          task,
          big_conv1_filters=FLAGS.big_conv1_filters,
          big_conv2_filters=FLAGS.big_conv2_filters,
          big_dense_size=FLAGS.big_dense_size,
          small_conv1_filters=FLAGS.small_conv1_filters,
          small_conv2_filters=FLAGS.small_conv2_filters,
          small_dense_size=FLAGS.small_dense_size)
    left_mask = [-1, -1, 0, -1, 1000, -1, 3, -1]
    right_mask = [0, 0, 1, 1, 3, 3, -1, -1]
  else:
    raise ValueError('task is unsupported')

  if FLAGS.shrink_unshrink_type == 'identity':
    make_shrink = shrink_unshrink_tff.make_identity_shrink
    make_unshrink = shrink_unshrink_tff.make_identity_unshrink
    server_model_fn = small_model_fn
    client_model_fn = small_model_fn
  elif FLAGS.shrink_unshrink_type == 'layerwise':
    make_shrink = shrink_unshrink_tff.make_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_layerwise_projection_unshrink
    server_model_fn = big_model_fn
    client_model_fn = small_model_fn
  elif FLAGS.shrink_unshrink_type == 'client_layerwise':
    make_shrink = shrink_unshrink_tff.make_client_specific_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_client_specific_layerwise_projection_unshrink
    server_model_fn = big_model_fn
    client_model_fn = small_model_fn
  elif FLAGS.shrink_unshrink_type == 'learned_layerwise':
    make_shrink = shrink_unshrink_tff.make_learned_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_learned_layerwise_projection_unshrink
    server_model_fn = big_model_fn
    client_model_fn = small_model_fn
  elif FLAGS.shrink_unshrink_type == 'learned_layerwise_v2':
    make_shrink = shrink_unshrink_tff.make_learnedv2_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_learned_layerwise_projection_unshrink
    server_model_fn = big_model_fn
    client_model_fn = small_model_fn
  elif FLAGS.shrink_unshrink_type == 'learned_sparse_layerwise_v2':
    make_shrink = shrink_unshrink_tff.make_learnedv2_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_learned_sparse_layerwise_projection_unshrink
    server_model_fn = big_model_fn
    client_model_fn = small_model_fn
  elif FLAGS.shrink_unshrink_type == 'static_client_layerwise':
    make_shrink = shrink_unshrink_tff.make_static_client_specific_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_client_specific_layerwise_projection_unshrink
    server_model_fn = big_model_fn
    client_model_fn = small_model_fn
  else:
    raise ValueError('invalid shrink unshrink passed')

  print('creating iterative process')

  if FLAGS.build_projection_matrix_type == 'normal':
    logging.info('using normal projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_normal_projection_matrix
  elif FLAGS.build_projection_matrix_type == 'dropout':
    logging.info('using dropout projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_dropout_projection_matrix
  elif FLAGS.build_projection_matrix_type == 'ensemble':
    logging.info('using ensemble projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_ensemble_dropout_projection_matrix
  elif FLAGS.build_projection_matrix_type == 'qr':
    logging.info('using qr projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_qr_projection_matrix
  elif FLAGS.build_projection_matrix_type == 'sparse':
    logging.info('using sparse projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_sparse_projection_matrix
  else:
    raise ValueError('invalid build_projection_matrix_type passed')
  shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
      left_mask=left_mask,
      right_mask=right_mask,
      build_projection_matrix=build_projection_matrix,
      new_projection_dict_decimate=FLAGS.new_projection_dict_decimate)

  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)

  if FLAGS.shrink_unshrink_type == 'static_client_layerwise':
    logging.info('static_client_layerwise iterative process')
    iterative_process = simple_fedavg_tff.build_federated_shrink_unshrink_process_with_client_id(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        client_id_to_dataset_preprocessor=train_data.dataset_computation,
        make_shrink=make_shrink,
        make_unshrink=make_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        oja_hyperparameter=FLAGS.oja_hyperparameter,
        static_client_layerwise_num_buckets=FLAGS
        .static_client_layerwise_num_buckets)
    training_process = iterative_process
  else:
    logging.info('standard iterative process')
    iterative_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        make_shrink=make_shrink,
        make_unshrink=make_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        oja_hyperparameter=FLAGS.oja_hyperparameter)
    training_process = (
        tff.simulation.compose_dataset_computation_with_iterative_process(
            train_data.dataset_computation, iterative_process))

  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=FLAGS.client_datasets_random_seed),
      size=FLAGS.clients_per_round)

  if task.datasets.validation_data is not None:
    validation_set = task.datasets.validation_data
  else:
    validation_set = task.datasets.test_data
  validation_set = validation_set.create_tf_dataset_from_all_clients()
  if FLAGS.num_validation_examples is not None:
    validation_set = validation_set.take(FLAGS.num_validation_examples)
  validation_set = task.datasets.eval_preprocess_fn(validation_set)
  federated_eval = tff.learning.build_federated_evaluation(server_model_fn)
  evaluation_selection_fn = lambda round_num: [validation_set]

  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model_weights, evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparam_flags()
  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  test_set = task.datasets.get_centralized_test_data()
  test_metrics = federated_eval(state.model_weights, [test_set])
  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
