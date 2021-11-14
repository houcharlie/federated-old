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

Specifically, we create (according to flags) an iterative processes that adapts
the client and server learning rate according to the history of loss values
encountered throughout training. For more details on the learning rate decay,
see `callbacks.py` and `adaptive_fed_avg.py`.
"""

from typing import Callable

from absl import app
from absl import flags
import tensorflow_federated as tff
import math
from adaptive_lr_decay import adaptive_fed_avg
from adaptive_lr_decay import callbacks
from optimization.cifar100 import federated_cifar100
from optimization.emnist import federated_emnist
from optimization.emnist_ae import federated_emnist_ae
from optimization.shakespeare import federated_shakespeare
from optimization.shared import optimizer_utils
from optimization.shared import training_specs
from optimization.stackoverflow import federated_stackoverflow
from optimization.stackoverflow_lr import federated_stackoverflow_lr
from utils import training_loop
from utils import utils_impl

_SUPPORTED_TASKS = [
    'cifar100', 'emnist_cr', 'emnist_ae', 'shakespeare', 'stackoverflow_nwp',
    'stackoverflow_lr'
]
with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as callback_flags:
  flags.DEFINE_float(
      'client_decay_factor', 0.1, 'Amount to decay the client learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'server_decay_factor', 0.9, 'Amount to decay the server learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'min_delta', 1e-4, 'Minimum delta for improvement in the learning rate '
      'callbacks.')
  flags.DEFINE_float(
      'switch_round', 0.1, 'Number of rounds before switching to minibatching ')
  flags.DEFINE_float(
      'swap_round', 0.1, 'Number of rounds before switching to minibatching in multistage ')
  flags.DEFINE_integer(
      'window_size', 100, 'Number of rounds to take a moving average over when '
      'estimating the training loss in learning rate callbacks.')
  flags.DEFINE_integer(
      'swapped', 0, 'Whether to start as minibatch ')
  flags.DEFINE_integer(
      'allow_swap', 1, 'Whether to swap in multistage ')
  flags.DEFINE_integer(
      'control', 0, 'Whether to use a control variate ')
  flags.DEFINE_integer(
      'multistage', 0, 'Whether to use multistage ')
  flags.DEFINE_integer(
      'patience', 100, 'Number of rounds of non-improvement before decaying the'
      'learning rate.')
  flags.DEFINE_float('min_lr', 0.0, 'The minimum learning rate.')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 20,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer('total_rounds', 500, 'Number of total training rounds.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/ocean/projects/iri180031p/houc/multistage/CIFAR-5core-longexp',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 10,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer(
      'rounds_per_profile', 0,
      '(Experimental) How often to run the experimental TF profiler, if >0.')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

  # CIFAR-100 flags
  flags.DEFINE_integer('cifar100_crop_size', 24, 'The height and width of '
                       'images after preprocessing.')

  # EMNIST CR flags
  flags.DEFINE_enum(
      'emnist_cr_model', 'cnn', ['cnn', '2nn'], 'Which model to '
      'use. This can be a convolutional model (cnn) or a two '
      'hidden-layer densely connected network (2nn).')

  # Shakespeare flags
  flags.DEFINE_integer(
      'shakespeare_sequence_length', 80,
      'Length of character sequences to use for the RNN model.')

  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer(
      'so_nwp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

  # Stack Overflow LR flags
  flags.DEFINE_integer('so_lr_vocab_tokens_size', 10000,
                       'Vocab tokens size used.')
  flags.DEFINE_integer('so_lr_vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer(
      'so_lr_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_lr_max_elements_per_user', 1000,
                       'Max number of training '
                       'sentences to use per user.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  # client_lr_callback = callbacks.create_reduce_lr_on_plateau(
  #     learning_rate=FLAGS.client_learning_rate,
  #     decay_factor=FLAGS.client_decay_factor,
  #     min_delta=FLAGS.min_delta,
  #     min_lr=FLAGS.min_lr,
  #     window_size=FLAGS.window_size,
  #     patience=FLAGS.patience)

  # server_lr_callback = callbacks.create_reduce_lr_on_plateau(
  #     learning_rate=FLAGS.server_learning_rate,
  #     decay_factor=FLAGS.server_decay_factor,
  #     min_delta=FLAGS.min_delta,
  #     min_lr=FLAGS.min_lr,
  #     window_size=FLAGS.window_size,
  #     patience=FLAGS.patience)
  #tff.backends.native.set_local_execution_context(FLAGS.clients_per_round)

#   client_lr_callback = callbacks.create_switch_lr(
#       owner='Client',
#       learning_rate=FLAGS.client_learning_rate,
#       start_lr=FLAGS.client_learning_rate,
#       decay_factor=0.,
#       switch_round=math.ceil(FLAGS.switch_round*FLAGS.total_rounds),
#       swapped=bool(FLAGS.swapped))
#   server_lr_callback = callbacks.create_switch_lr(
#       owner='Server',
#       learning_rate=FLAGS.server_learning_rate,
#       start_lr=FLAGS.server_learning_rate,
#       decay_factor=1.,
#       switch_round=math.ceil(FLAGS.switch_round*FLAGS.total_rounds),
#       swapped=bool(FLAGS.swapped))
  if FLAGS.multistage == 1:
    client_lr_callback = callbacks.create_multistage_lr(
        owner='Client',
        learning_rate=FLAGS.client_learning_rate,
        start_lr=FLAGS.client_learning_rate,
        s=0.,
        decay_factor=1.,
        total_rounds=FLAGS.total_rounds,
        rounds_in_stage=0,
        swapped=bool(FLAGS.swapped),
        sampled_clients=FLAGS.clients_per_round,
        switch_round=FLAGS.switch_round*FLAGS.total_rounds,
        swap_round=math.ceil(FLAGS.swap_round*FLAGS.total_rounds))
    server_lr_callback = callbacks.create_multistage_lr(
        owner='Server',
        learning_rate=FLAGS.server_learning_rate,
        start_lr=FLAGS.client_learning_rate,
        s=0.,
        decay_factor=1.,
        total_rounds=FLAGS.total_rounds,
        rounds_in_stage=0,
        swapped=bool(FLAGS.swapped),
        sampled_clients=FLAGS.clients_per_round,
        switch_round=FLAGS.switch_round*FLAGS.total_rounds,
        swap_round=math.ceil(FLAGS.swap_round*FLAGS.total_rounds))
  elif FLAGS.multistage == 0:
    client_lr_callback = callbacks.create_switch_lr(
        owner='Client',
        learning_rate=FLAGS.client_learning_rate,
        start_lr=FLAGS.client_learning_rate,
        decay_factor=1.,
        switch_round=math.ceil(FLAGS.switch_round*FLAGS.total_rounds),
        swapped=bool(FLAGS.swapped))
    server_lr_callback = callbacks.create_switch_lr(
        owner='Server',
        learning_rate=FLAGS.server_learning_rate,
        start_lr=FLAGS.server_learning_rate,
        decay_factor=1.,
        switch_round=math.ceil(FLAGS.switch_round*FLAGS.total_rounds),
        swapped=bool(FLAGS.swapped))
  elif FLAGS.multistage == 2:
    client_lr_callback = callbacks.create_constantstage_lr(
        owner='Client',
        learning_rate=FLAGS.client_learning_rate,
        start_lr=FLAGS.client_learning_rate,
        s=0.,
        decay_factor=0.,
        total_rounds=FLAGS.total_rounds,
        rounds_in_stage=0,
        swapped=bool(FLAGS.swapped),
        sampled_clients=FLAGS.clients_per_round,
        switch_round=FLAGS.switch_round*FLAGS.total_rounds,
        allow_swap=bool(FLAGS.allow_swap))
    server_lr_callback = callbacks.create_constantstage_lr(
        owner='Server',
        learning_rate=FLAGS.server_learning_rate,
        start_lr=FLAGS.client_learning_rate,
        s=0.,
        decay_factor=1.,
        total_rounds=FLAGS.total_rounds,
        rounds_in_stage=0,
        swapped=bool(FLAGS.swapped),
        sampled_clients=FLAGS.clients_per_round,
        switch_round=FLAGS.switch_round*FLAGS.total_rounds,
        allow_swap=bool(FLAGS.allow_swap))

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    return adaptive_fed_avg.build_fed_avg_process(
        model_fn,
        client_lr_callback,
        server_lr_callback,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        control=bool(FLAGS.control))

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  if FLAGS.task == 'cifar100':
    runner_spec = federated_cifar100.configure_training(
        task_spec, crop_size=FLAGS.cifar100_crop_size)
  elif FLAGS.task == 'emnist_cr':
    runner_spec = federated_emnist.configure_training(
        task_spec, model=FLAGS.emnist_cr_model)
  elif FLAGS.task == 'emnist_ae':
    runner_spec = federated_emnist_ae.configure_training(task_spec)
  elif FLAGS.task == 'shakespeare':
    runner_spec = federated_shakespeare.configure_training(
        task_spec, sequence_length=FLAGS.shakespeare_sequence_length)
  elif FLAGS.task == 'stackoverflow_nwp':
    runner_spec = federated_stackoverflow.configure_training(
        task_spec,
        vocab_size=FLAGS.so_nwp_vocab_size,
        num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
        sequence_length=FLAGS.so_nwp_sequence_length,
        max_elements_per_user=FLAGS.so_nwp_max_elements_per_user,
        num_validation_examples=FLAGS.so_nwp_num_validation_examples)
  elif FLAGS.task == 'stackoverflow_lr':
    runner_spec = federated_stackoverflow_lr.configure_training(
        task_spec,
        vocab_tokens_size=FLAGS.so_lr_vocab_tokens_size,
        vocab_tags_size=FLAGS.so_lr_vocab_tags_size,
        max_elements_per_user=FLAGS.so_lr_max_elements_per_user,
        num_validation_examples=FLAGS.so_lr_num_validation_examples)
  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))

  training_loop.run(
      iterative_process=runner_spec.iterative_process,
      client_ids=runner_spec.client_ids,
      client_datasets_fn=runner_spec.client_datasets_fn,
      validation_fn=runner_spec.validation_fn,
      train_eval_fn=runner_spec.train_eval_fn,
      test_fn=runner_spec.test_fn,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      rounds_per_profile=FLAGS.rounds_per_profile)


if __name__ == '__main__':
  app.run(main)
