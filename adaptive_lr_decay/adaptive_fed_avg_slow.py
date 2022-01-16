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
"""Interpolation between FedAvg and FedSGD with adaptive learning rate decay.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Any, Callable, Optional, Union, Tuple

import attr
import tensorflow as tf
import tensorflow_federated as tff

from adaptive_lr_decay import callbacks

# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[[Any], float]
Callback = Union[callbacks.ReduceLROnPlateau, callbacks.SwitchLR, callbacks.MultistageLR]


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')


def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights(
      trainable=tuple(model.trainable_variables),
      non_trainable=tuple(model.non_trainable_variables))


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Attributes:
    model: A dictionary of the model's trainable and non-trainable weights.
    optimizer_state: The server optimizer variables.
    client_lr_callback: A `callback.LROnPlateau` instance.
    server_lr_callback: A `callback.LROnPlateau` instance.
  """
  model = attr.ib()
  model_ghost_fedavg = attr.ib()
  model_ghost_mbsgd = attr.ib()
  optimizer_state = attr.ib()
  client_lr_callback = attr.ib()
  server_lr_callback = attr.ib()

@attr.s(eq=False, order=False, frozen=True)
class ClientState(object):
  """Structure for state on the server.

  Attributes:
    correction: The correction applied to client steps (control variate)
  """
  correction = attr.ib()

@tf.function
def update_callback(server_state, num_client_grads, client_monitor_value, server_monitor_value):
  updated_client_lr_callback = server_state.client_lr_callback.update(
      client_monitor_value, num_client_grads)
  updated_server_lr_callback = server_state.server_lr_callback.update(
      server_monitor_value, num_client_grads)
  return updated_client_lr_callback, updated_server_lr_callback
@tf.function
def server_update(model, model_fedavg, model_mbsgd, server_optimizer, server_optimizer_fedavg, server_optimizer_mbsgd, server_state, aggregated_gradients, fedavg_update, sgd_update,
                  client_monitor_value, server_monitor_value, num_client_grads,
                  updated_client_lr_callback, updated_server_lr_callback):
  """Updates `server_state` according to `weights_delta` and output metrics.

  The `model_weights` attribute of `server_state` is updated according to the
  ``weights_delta`. The `client_lr_callback` and `server_lr_callback` attributes
  are updated according to the `client_monitor_value` and `server_monitor_value`
  arguments, respectively.

  Args:
    model: A `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    aggregated_gradients: A weighted average over clients of the per-client
      gradient sums.
    client_monitor_value: The updated round metric used to update the client
      callback.
    server_monitor_value: The updated round metric used to update the server
      callback.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

  # Apply the update to the model. Note that we do not multiply by -1.0, since
  # we actually accumulate the client gradients.
  grads_and_vars = [
      (x, v) for x, v in zip(aggregated_gradients, model_weights.trainable)
  ]

  server_optimizer.apply_gradients(grads_and_vars)

  model_ghost_fedavg = _get_weights(model_fedavg)
  tff.utils.assign(model_ghost_fedavg, server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tff.utils.assign(server_optimizer_fedavg.variables(), server_state.optimizer_state)

  # Apply the update to the model. Note that we do not multiply by -1.0, since
  # we actually accumulate the client gradients.
  grads_and_vars = [
      (x, v) for x, v in zip(fedavg_update, model_ghost_fedavg.trainable)
  ]

  server_optimizer_fedavg.apply_gradients(grads_and_vars)

  model_ghost_mbsgd = _get_weights(model_mbsgd)
  tff.utils.assign(model_ghost_mbsgd, server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tff.utils.assign(server_optimizer_mbsgd.variables(), server_state.optimizer_state)

  # Apply the update to the model. Note that we do not multiply by -1.0, since
  # we actually accumulate the client gradients.
  grads_and_vars = [
      (x, v) for x, v in zip(sgd_update, model_ghost_mbsgd.trainable)
  ]

  server_optimizer_mbsgd.apply_gradients(grads_and_vars)


  

  # updated_client_lr_callback = server_state.client_lr_callback.update(
  #     client_monitor_value, num_client_grads)
  # updated_server_lr_callback = server_state.server_lr_callback.update(
  #     server_monitor_value, num_client_grads)

  # Create a new state based on the updated model.
  return tff.utils.update_state(
      server_state,
      model=model_weights,
      model_ghost_fedavg=model_ghost_fedavg,
      model_ghost_mbsgd=model_ghost_mbsgd,
      optimizer_state=server_optimizer.variables(),
      client_lr_callback=updated_client_lr_callback,
      server_lr_callback=updated_server_lr_callback)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Attributes:
    accumulated_gradients: A list of accumulated gradients for the model's
      trainable variables. Note: This is a sum of gradients, not the difference
      between the initial brodcast model, and the trained model (as in
      `tff.learning.build_federated_averaging_process`).
    client_weight: Weight to be used in a weighted mean when aggregating
      the `weights_delta`.
    initial_model_output: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      evaluating on the input dataset (before training).
    model_output: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
    optimizer_output: Additional metrics or other outputs defined by the
      optimizer.
  """
  accumulated_gradients = attr.ib()
  accumulated_gradients_nonupdate = attr.ib()
  client_weight = attr.ib()
  num_grads = attr.ib()
  initial_model_output = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


@tf.function
def get_client_output(model, dataset, weights):
  """Evaluates the metrics of a client model."""
  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, weights)
  for batch in dataset:
    model.forward_pass(batch)
  return model.report_local_outputs()


@tf.function
def client_update(model,
                  dataset,
                  initial_weights,
                  client_optimizer,
                  correction,
                  client_weight_fn=None):
  """Updates the client model with local training.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    initial_weights: A `tff.learning.Model.weights` from server.
    client_optimizer: A `tf.keras.optimizer.Optimizer` object.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A 'ClientOutput`.
  """

  model_weights = _get_weights(model)
  tff.utils.assign(model_weights, initial_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  num_grads = tf.constant(0., dtype=tf.float32)
  grad_sums = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  grad_sums_sgd = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  # compute the sgd update
  for batch in dataset:
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch)
    grads = tape.gradient(output.loss, model_weights.trainable)
    grad_sums_sgd = tf.nest.map_structure(tf.add, grad_sums_sgd, grads)

  # compute the fedavg updates
  for batch in dataset:
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch)
    grads = tape.gradient(output.loss, model_weights.trainable)
    corrected_grads = tf.nest.map_structure(lambda a,b: a + b,
                                            grads, correction)
    grads_and_vars = zip(corrected_grads, model_weights.trainable)
    client_optimizer.apply_gradients(grads_and_vars)
    num_examples += tf.shape(output.predictions)[0]
    num_grads += 1.
    grad_sums = tf.nest.map_structure(tf.add, grad_sums, grads)

  aggregated_outputs = model.report_local_outputs()

  if client_weight_fn is None:
    client_weight = tf.cast(num_examples, dtype=tf.float32)
  else:
    client_weight = client_weight_fn(aggregated_outputs)

  return ClientOutput(
      accumulated_gradients=grad_sums,
      accumulated_gradients_nonupdate=grad_sums_sgd,
      client_weight=client_weight,
      num_grads=num_grads,
      initial_model_output=aggregated_outputs,
      model_output=aggregated_outputs,
      optimizer_output=collections.OrderedDict([('num_examples', num_examples)
                                               ]))


def build_server_init_fn(model_fn: ModelBuilder,
                         server_optimizer_fn: OptimizerBuilder,
                         client_lr_callback: Callback,
                         server_lr_callback: Callback):
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    client_lr_callback: A `ReduceLROnPlateau` callback.
    server_lr_callback: A `ReduceLROnPlateau` callback.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    server_optimizer = server_optimizer_fn(server_lr_callback.learning_rate)
    model = model_fn()
    model_ghost_mbsgd = model_fn()
    model_ghost_fedavg = model_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=_get_weights(model),
        model_ghost_fedavg=_get_weights(model_ghost_fedavg),
        model_ghost_mbsgd=_get_weights(model_ghost_mbsgd),
        optimizer_state=server_optimizer.variables(),
        client_lr_callback=client_lr_callback,
        server_lr_callback=server_lr_callback)

  return server_init_tf

def build_client_init_fn(model_fn: ModelBuilder):
  """Builds a `tff.tf_computation` that returns the initial `ClientState`.

  The attribute `ClientState.correction` is initialized by ModelBuilder

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.

  Returns:
    A `tff.tf_computation` that returns initial `ClientState`.
  """

  @tff.tf_computation
  def client_init_tf():
    model = model_fn()
    # the default value for the correction is zero.
    return ClientState(
        correction=tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable))

  return client_init_tf

def build_fed_avg_process(
    model_fn: ModelBuilder,
    client_lr_callback: Callback,
    server_lr_callback: Callback,
    client_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    client_weight_fn: Optional[ClientWeightFn] = None,
    control: bool = False
):
  """Builds the TFF computations for FedAvg with learning rate decay.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_lr_callback: A `ReduceLROnPlateau` callback.
    server_lr_callback: A `ReduceLROnPlateau` callback.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  dummy_model = model_fn()
  client_monitor = client_lr_callback.monitor
  server_monitor = server_lr_callback.monitor

  server_init_tf = build_server_init_fn(model_fn, server_optimizer_fn,
                                        client_lr_callback, server_lr_callback)
  client_init_tf = build_client_init_fn(model_fn)

  server_state_type = server_init_tf.type_signature.result
  client_state_type = client_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  
  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)

  client_lr_type = server_state_type.client_lr_callback.learning_rate
  swapped_type = server_state_type.client_lr_callback.swapped
  client_monitor_value_type = server_state_type.client_lr_callback.best
  server_monitor_value_type = server_state_type.server_lr_callback.best

  @tff.tf_computation(model_input_type, model_weights_type, client_lr_type, client_state_type)
  def client_update_fn(tf_dataset, initial_model_weights, client_lr, client_state):
    client_optimizer = client_optimizer_fn(client_lr)
    initial_model_output = get_client_output(model_fn(), tf_dataset,
                                             initial_model_weights)
    client_output = client_update(model_fn(), tf_dataset, initial_model_weights,
                                 client_optimizer, client_state.correction, client_weight_fn)
    return tff.utils.update_state(
        client_output, initial_model_output=initial_model_output)

  @tff.tf_computation(server_state_type, model_weights_type.trainable, model_weights_type.trainable, model_weights_type.trainable,
                      client_monitor_value_type, server_monitor_value_type,
                      tf.float32)
  def server_update_fn(server_state, model_delta, fedavg_delta, sgd_delta, client_monitor_value,
                       server_monitor_value, num_client_grads):
    model = model_fn()
    fedavg_model = model_fn()
    mbsgd_model = model_fn()
    updated_client_callback, updated_server_callback = update_callback(server_state, num_client_grads,
                                                                       client_monitor_value, server_monitor_value)
    server_lr = server_state.server_lr_callback.learning_rate
    server_optimizer = server_optimizer_fn(server_lr)
    server_optimizer_fedavg = server_optimizer_fn(server_lr)
    server_optimizer_mbsgd = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    _initialize_optimizer_vars(fedavg_model, server_optimizer_fedavg)
    _initialize_optimizer_vars(mbsgd_model, server_optimizer_mbsgd)
    return server_update(model, fedavg_model, mbsgd_model, server_optimizer, server_optimizer_fedavg, server_optimizer_mbsgd, server_state, model_delta, fedavg_delta, sgd_delta,
                         client_monitor_value, server_monitor_value, num_client_grads,
                         updated_client_callback, updated_server_callback)
  client_output_type = client_update_fn.type_signature.result

  @tff.tf_computation(model_weights_type.trainable, model_weights_type.trainable, swapped_type)
  def choose_aggregation(aggregated_gradients_fedavg, aggregated_gradients_sgd, swapped):
    tf.print('Swap status:', swapped)
    return tf.cond(swapped,
                    lambda: aggregated_gradients_sgd,
                    lambda: aggregated_gradients_fedavg)
  @tff.federated_computation(
      tff.type_at_server(server_state_type), 
      tff.type_at_clients(client_state_type),
      tff.type_at_clients(tf_dataset_type))
  def run_one_round(server_state, client_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Note that in addition to updating the server weights according to the client
    model weight deltas, we extract metrics (governed by the `monitor` attribute
    of the `client_lr_callback` and `server_lr_callback` attributes of the
    `server_state`) and use these to update the client learning rate callbacks.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation` before and during local
      client training.
    """
    # server_state, client_state = state
    # #sampled_client_state = client_state
    client_model = tff.federated_broadcast(server_state.model)
    client_lr = tff.federated_broadcast(
        server_state.client_lr_callback.learning_rate)

    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, client_model, client_lr, client_state))

    # c is the minibatch gradient across all clients
    @tff.tf_computation(model_weights_type.trainable, tf.float32)
    def compute_c(aggregated_gradients_sgd, num_client_grads):
      c = tf.nest.map_structure(lambda a: a/num_client_grads, aggregated_gradients_sgd)
      return c

    client_weight = client_outputs.client_weight
    num_client_grads = tff.federated_sum(client_outputs.num_grads)
    aggregated_gradients_fedavg = tff.federated_mean(
          client_outputs.accumulated_gradients, weight=client_weight)
    aggregated_gradients_nonupdates = client_outputs.accumulated_gradients_nonupdate
    sgd_grad_sum = tff.federated_sum(aggregated_gradients_nonupdates)
    # the sgd sampled gradient at the current iterate
    sgd_grad = tff.federated_map(compute_c, (sgd_grad_sum, num_client_grads))
  

    aggregated_gradients_sgd = tff.federated_sum(
          client_outputs.accumulated_gradients)
    c = tff.federated_broadcast(tff.federated_map(compute_c, (aggregated_gradients_sgd, num_client_grads)))
    # c_i is the minibatch gradient within one client
    @tff.tf_computation(client_output_type.accumulated_gradients, client_output_type.num_grads)
    def compute_ci(accumulated_gradients, num_grads):
      c_i = tf.nest.map_structure(lambda a: a/num_grads, accumulated_gradients)
      return c_i
    c_i = tff.federated_map(compute_ci, (client_outputs.accumulated_gradients, client_outputs.num_grads))

    @tff.tf_computation(client_output_type.accumulated_gradients, client_output_type.accumulated_gradients)
    def compute_control_input(c, c_i):
      correction = tf.nest.map_structure(lambda a, b: a - b, c, c_i)
      # if we are using SCAFFOLD then use the correction, otherwise 
      # we just let the correction be zero.
      return tf.cond(tf.constant(control, dtype=tf.bool), 
          lambda: correction, 
          lambda: tf.nest.map_structure(tf.zeros_like, correction))
    
    corrections = tff.federated_map(compute_control_input, (c, c_i))
    @tff.tf_computation(client_state_type, client_output_type.accumulated_gradients)
    def update_client_state(client_state, corrections):
      return tff.utils.update_state(
                    client_state,
                    correction=corrections)
    client_state = tff.federated_map(update_client_state, (client_state, corrections))

    aggregated_gradients = tff.federated_map(choose_aggregation, 
                                             (aggregated_gradients_fedavg, 
                                              sgd_grad,
                                              server_state.server_lr_callback.swapped)
                                            )
    # pseudogradient norm measurement
    @tff.tf_computation(model_weights_type.trainable)
    def calculate_global_norm(model_update):
      """Calculate the global norm across all layers of the model update."""
      return tf.linalg.global_norm(tf.nest.flatten(model_update))
    
    @tff.tf_computation(model_weights_type.trainable, swapped_type, tf.float32)
    def choose_which_norm(aggregated_gradients, swapped, num_gradients):
      return tf.cond(swapped,
                      lambda: tf.nest.map_structure(lambda x: x/num_gradients, aggregated_gradients),
                      lambda: aggregated_gradients)
    proper_normalized_server_update = tff.federated_map(choose_which_norm, (aggregated_gradients, server_state.server_lr_callback.swapped, num_client_grads))
    server_update_norm = tff.federated_map(calculate_global_norm, proper_normalized_server_update)

    
    initial_aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.initial_model_output)
    if isinstance(initial_aggregated_outputs.type_signature, tff.StructType):
      initial_aggregated_outputs = tff.federated_zip(initial_aggregated_outputs)

    aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.model_output)
    if isinstance(aggregated_outputs.type_signature, tff.StructType):
      aggregated_outputs = tff.federated_zip(aggregated_outputs)
    client_monitor_value = initial_aggregated_outputs[client_monitor]
    server_monitor_value = initial_aggregated_outputs[server_monitor]

    server_state = tff.federated_map(
        server_update_fn, (server_state, aggregated_gradients,aggregated_gradients_fedavg,sgd_grad,
                           client_monitor_value, server_monitor_value,
                           num_client_grads))

    

    @tff.tf_computation(model_weights_type.trainable)
    def normalize_vector(model_update):
      update_norm = calculate_global_norm(model_update)
      return tf.nest.map_structure(lambda a: a / update_norm, model_update)

    @tff.tf_computation(model_weights_type.trainable)
    def calculate_square_global_norm(model_update):
      """Calculate the squared global norm across all layers of a model update."""
      # We compute this directly in order to circumvent precision issues
      # incurred by taking square roots and then re-squaring.
      return calculate_global_norm(model_update)**2

    @tff.tf_computation(tf.float32, tf.float32)
    def compute_average_cosine_similarity(square_norm_of_sum, num_vectors):
      """Calculate the average cosine similarity between unit length vectors.
      Args:
        square_norm_of_sum: The squared norm of the sum of the normalized vectors.
        num_vectors: The number of vectors the sum was taken over.
      Returns:
        A float representing the average pairwise cosine similarity among all
          vectors.
      """
      return (square_norm_of_sum - num_vectors) / (
          num_vectors * (num_vectors - 1.0))
    
    # compute pairwise average cosine similarity
    normalized_updates = tff.federated_map(normalize_vector, client_outputs.accumulated_gradients)
    sum_of_normalized_updates = tff.federated_sum(normalized_updates)
    square_norm_of_sum = tff.federated_map(calculate_square_global_norm,
                                            sum_of_normalized_updates)
    num_clients = tff.federated_sum(tff.federated_value(1.0, tff.CLIENTS))
    average_cosine_similarity = tff.federated_map(
        compute_average_cosine_similarity, (square_norm_of_sum, num_clients))
    
    # compute cosine similarity between fedavg update and sgd update
    sgd_grad_normalized = tff.federated_map(normalize_vector, sgd_grad)
    server_update_normalized = tff.federated_map(normalize_vector, aggregated_gradients)

    @tff.tf_computation(model_weights_type.trainable, model_weights_type.trainable)
    def dot_product(v1, v2):
      flatv1 = tf.nest.map_structure(lambda v: tf.squeeze(tf.reshape(v, [1,-1])), v1)
      flatv2 = tf.nest.map_structure(lambda v: tf.squeeze(tf.reshape(v, [1,-1])), v2)
      dots = tf.nest.map_structure(lambda u, w: tf.tensordot(u, w, 1), flatv1, flatv2)
      dot = tf.reduce_sum(tf.nest.flatten(dots))
      return dot

    sgd_vs_serverupdate = tff.federated_map(dot_product, (sgd_grad_normalized, server_update_normalized))
    
    # compute average cosine similarity over client pseudogradients and sgd update
    sgd_grad_broadcast = tff.federated_broadcast(sgd_grad_normalized)
    sgd_client_pseudo_dots = tff.federated_map(dot_product, (sgd_grad_broadcast, normalized_updates))
    sgd_client_pseudo_mean = tff.federated_mean(sgd_client_pseudo_dots)

    # compute average cosine similarity between client gradients and sgd update
    normalized_updates_sgd = tff.federated_map(normalize_vector, client_outputs.accumulated_gradients_nonupdate)
    sgd_client_true_dots = tff.federated_map(dot_product, (sgd_grad_broadcast, normalized_updates_sgd))
    sgd_client_true_mean = tff.federated_mean(sgd_client_true_dots)

    
    sgd_gradient_norm = tff.federated_map(calculate_global_norm, sgd_grad)
    fedavg_norm = tff.federated_map(calculate_global_norm, aggregated_gradients_fedavg)

    result = collections.OrderedDict(
        before_training=initial_aggregated_outputs,
        during_training=aggregated_outputs,
        average_cosine_similarity=average_cosine_similarity,
        sgd_vs_serverupdate=sgd_vs_serverupdate,
        sgd_client_pseudo_mean=sgd_client_pseudo_mean,
        sgd_client_true_mean=sgd_client_true_mean,
        server_update_norm=server_update_norm,
        sgd_gradient_norm=sgd_gradient_norm,
        fedavg_norm=fedavg_norm
        )

    return server_state, client_state, result

  @tff.federated_computation
  def initialize_fn():
    server_state = tff.federated_value(server_init_tf(), tff.SERVER)
    #client_state = tff.federated_value(client_init_tf(), tff.CLIENTS)
    return server_state

  iterative_process = tff.templates.IterativeProcess(
      initialize_fn=initialize_fn, next_fn=run_one_round)

  @tff.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  @tff.tf_computation(server_state_type)
  def get_model_fedavg_weights(server_state):
    return server_state.model_ghost_fedavg

  @tff.tf_computation(server_state_type)
  def get_model_mbsgd_weights(server_state):
    return server_state.model_ghost_mbsgd
      
  @tff.tf_computation
  def client_init():
    return client_init_tf()
  iterative_process.get_model_weights = get_model_weights
  iterative_process.get_model_fedavg_weights = get_model_fedavg_weights
  iterative_process.get_model_mbsgd_weights = get_model_mbsgd_weights
  iterative_process.client_init = client_init
  return iterative_process
