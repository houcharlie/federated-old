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
"""TensorFlow training code for Federated GANs.

This code is intended to only use vanilla TensorFlow (no TFF dependency); it is
wired together into a federated computation in gan_training_tff_fns.py. The one
exception is some handling for conversion from Struct, which should go
away when b/130724878 is fixed.
"""

import collections

import attr
import tensorflow as tf
import tensorflow_federated as tff
from gans import gan_losses
from utils import tensor_utils


def assert_no_anon_tuples(x):
  """Checks that a nested structure has no Structs at the leaves."""

  def check_anon(t):
    if 'Struct' in str(type(t)):
      raise ValueError('Found Struct:\n', t)
    return None

  tf.nest.map_structure(check_anon, x)
  return x


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=True)
class FromServer(object):
  """Container for data that is broadcast from the server to clients.

  Attributes:
    generator_weights: Weights for the generator model, in the order of
      `tf.keras.Model.weights`.
    discriminator_weights: Weights for the discriminator model, in the order of
      `tf.keras.Model.weights`.
  """
  generator_weights = attr.ib()
  discriminator_weights = attr.ib()
  meta_gen = attr.ib()
  meta_disc = attr.ib()


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=False)
class ServerState(object):
  """Container for all server state that must pass from round to round.

  Attributes:
    generator_weights: Weights for the generator model, in the order of
      `tf.keras.Model.weights`.
    discriminator_weights: Weights for the discriminator model, in the order of
      `tf.keras.Model.weights`.
    counters: Aggregated training counters.
    aggregation_state: State of the aggregation process. This aggregation
      process could be used to handle Differential Privacy aggregation, or could
      be set to a simple stateless mean for a non-Differentially Private
      approach.
  """
  generator_weights = attr.ib()
  discriminator_weights = attr.ib()
  meta_gen = attr.ib()
  meta_disc = attr.ib()
  counters = attr.ib()
  aggregation_state = attr.ib(default=())


# Set cmp=False to get a default hash function for tf.function.
@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Container for data that is sent from clients back to the server..

  Attributes:
    discriminator_weights_delta: Update for the discriminator model, in the
      order of tf.keras.Model.weights`.
    update_weight: Weight to be associated with the update.
    counters: Metrics that are summed across clients.
  """
  discriminator_weights_delta = attr.ib()
  generator_weights_delta = attr.ib()
  update_weight = attr.ib()
  counters = attr.ib()


def _weights(model):
  """Returns tensors of model weights, in the order of the variables."""
  return [v.read_value() for v in model.weights]

@tf.function
def client_computation(
    # Tensor/Dataset arguments that will be supplied by TFF:
    gen_inputs_ds: tf.data.Dataset,
    real_data_ds: tf.data.Dataset,
    from_server: FromServer,
    # Python arguments bound to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    control_input_gen: tf.keras.Model.weights,
    control_input_disc: tf.keras.Model.weights,
    tau: float) -> ClientOutput:
  """The computation to run on the client, training the discriminator.

  Args:
    gen_inputs_ds: A `tf.data.Dataset` of generator_inputs.
    real_data_ds: A `tf.data.Dataset` of data from the real distribution.
    from_server: A `FromServer` object, including the current model weights.
    generator:  The generator.
    discriminator: The discriminator.
    train_discriminator_fn: A function which takes the two networks, generator
      input, and real data and trains the discriminator.

  Returns:
    A `ClientOutput` object.
  """
  tf.nest.map_structure(lambda a, b: a.assign(b), generator.weights,
                        from_server.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), discriminator.weights,
                        from_server.discriminator_weights)
  meta_gen = from_server.meta_gen
  meta_disc = from_server.meta_disc
  num_examples = tf.constant(0)
  gen_inputs_and_real_data = tf.data.Dataset.zip((gen_inputs_ds, real_data_ds))
  loss_fns = gan_losses.WassersteinGanLossFns()
  for gen_inputs, real_data in gen_inputs_and_real_data:
    # It's possible that real_data and gen_inputs have different batch sizes.
    # For calculating the discriminator loss, it's desirable to have equal-sized
    # contributions from both the real and fake data. Also, it's necessary if
    # using the Wasserstein gradient penalty (where a difference is taken b/w
    # the real and fake data). So here we reduce to the min batch size. This
    # also ensures num_examples properly reflects the amount of data trained on.
    min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(gen_inputs)[0])
    real_data = real_data[0:min_batch_size]
    gen_inputs = gen_inputs[0:min_batch_size]
    with tf.GradientTape() as tape_gen:
      gen_loss = loss_fns.generator_loss(generator, discriminator, gen_inputs)
      for i in range(len(generator.weights)):
        gen_loss += tau*tf.nn.l2_loss(generator.weights[i] - meta_gen[i])
    with tf.GradientTape() as tape_disc:
      disc_loss = loss_fns.discriminator_loss(generator, discriminator, gen_inputs, 
                                            real_data)     
      for i in range(len(discriminator.weights)):
        disc_loss += tau*tf.nn.l2_loss(discriminator.weights[i] - meta_disc[i])     
    # apply the gradient                        
    disc_grads = tape_disc.gradient(disc_loss, discriminator.weights)
    disc_grads_and_vars = zip(disc_grads, discriminator.weights)
    gen_grads = tape_gen.gradient(gen_loss, generator.weights)
    gen_grads_and_vars = zip(gen_grads, generator.weights)
    disc_optimizer.apply_gradients(disc_grads_and_vars)
    gen_optimizer.apply_gradients(gen_grads_and_vars)

    # apply the adjustment
    disc_adj_gv = tf.nest.map_structure(lambda x,v: (-1.0*x,v), control_input_disc, 
                                        discriminator.weights)
    gen_adj_gv = tf.nest.map_structure(lambda x,v: (-1.0*x,v), control_input_gen,
                                        generator.weights)

    disc_optimizer.apply_gradients(disc_adj_gv)
    gen_optimizer.apply_gradients(gen_adj_gv)
    num_examples += min_batch_size

  disc_delta = tf.nest.map_structure(tf.subtract, discriminator.weights,
                                        from_server.discriminator_weights)
  gen_delta = tf.nest.map_structure(tf.subtract, generator.weights,
                                        from_server.generator_weights)
  disc_delta, disc_has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(disc_delta))
  gen_delta, gen_has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(gen_delta))   
  update_weight = tf.cast(num_examples, tf.float32)
  # Zero out the weight if there are any non-finite values.
  # TODO(b/122071074): federated_mean might not do the right thing if
  # all clients have zero weight.
  update_weight_disc = tf.cond(
      tf.equal(disc_has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))
  update_weight_gen = tf.cond(
      tf.equal(gen_has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))   
  update_weight = tf.math.minimum(update_weight_disc, update_weight_gen) 
  return ClientOutput(
      discriminator_weights_delta=disc_delta,
      generator_weights_delta=gen_delta,
      update_weight=update_weight,
      counters=collections.OrderedDict(
          num_discriminator_train_examples=num_examples))

@tf.function
def client_control(
    # Tensor/Dataset arguments that will be supplied by TFF:
    gen_inputs_ds: tf.data.Dataset,
    real_data_ds: tf.data.Dataset,
    from_server: FromServer,
    # Python arguments bound to be bound at TFF computation construction time:
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    zero_disc: tf.keras.Model,
    zero_gen: tf.keras.Model,
    tau: float) -> ClientOutput:
  """The computation to run on the client, training the discriminator.

  Args:
    gen_inputs_ds: A `tf.data.Dataset` of generator_inputs.
    real_data_ds: A `tf.data.Dataset` of data from the real distribution.
    from_server: A `FromServer` object, including the current model weights.
    generator:  The generator.
    discriminator: The discriminator.
    train_discriminator_fn: A function which takes the two networks, generator
      input, and real data and trains the discriminator.

  Returns:
    A `ClientOutput` object.
  """
  tf.nest.map_structure(lambda a, b: a.assign(b), generator.weights,
                        from_server.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), discriminator.weights,
                        from_server.discriminator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), zero_gen.weights,
                        from_server.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), zero_disc.weights,
                        from_server.discriminator_weights)
  num_examples = tf.constant(0)
  meta_gen = from_server.meta_gen
  meta_disc = from_server.meta_disc
  gen_inputs_and_real_data = tf.data.Dataset.zip((gen_inputs_ds, real_data_ds))
  loss_fns = gan_losses.WassersteinGanLossFns()
  for gen_inputs, real_data in gen_inputs_and_real_data:
    # It's possible that real_data and gen_inputs have different batch sizes.
    # For calculating the discriminator loss, it's desirable to have equal-sized
    # contributions from both the real and fake data. Also, it's necessary if
    # using the Wasserstein gradient penalty (where a difference is taken b/w
    # the real and fake data). So here we reduce to the min batch size. This
    # also ensures num_examples properly reflects the amount of data trained on.
    min_batch_size = tf.minimum(tf.shape(real_data)[0], tf.shape(gen_inputs)[0])
    real_data = real_data[0:min_batch_size]
    gen_inputs = gen_inputs[0:min_batch_size]
    # reset the gen/discriminator values so there's no moving
    #tf.nest.map_structure(lambda a, b: a.assign(b), generator.weights,
                        #from_server.generator_weights)
    #tf.nest.map_structure(lambda a, b: a.assign(b), discriminator.weights,
                          #from_server.discriminator_weights)
    with tf.GradientTape() as tape_gen:
      gen_loss = loss_fns.generator_loss(generator, discriminator, gen_inputs)
      for i in range(len(generator.weights)):
        gen_loss += tau*tf.nn.l2_loss(generator.weights[i] - meta_gen[i])
    with tf.GradientTape() as tape_disc:
      disc_loss = loss_fns.discriminator_loss(generator, discriminator, gen_inputs, 
                                            real_data)     
      for i in range(len(discriminator.weights)):
        disc_loss += tau*tf.nn.l2_loss(discriminator.weights[i] - meta_disc[i])
    # get disc grads
    disc_grads = tape_disc.gradient(disc_loss, discriminator.weights)
    disc_grads_and_vars = zip(disc_grads, discriminator.weights)

    # get gen grads
    gen_grads = tape_gen.gradient(gen_loss, generator.weights)
    gen_grads_and_vars = zip(gen_grads, generator.weights)
    
    disc_grads_and_vars = tf.nest.map_structure(lambda x,v: (x,v), disc_grads, 
                                        zero_disc.weights)
    gen_grads_and_vars = tf.nest.map_structure(lambda x,v: (x,v), gen_grads,
                                        zero_gen.weights)
    #apply the gradients
    disc_optimizer.apply_gradients(disc_grads_and_vars)
    gen_optimizer.apply_gradients(gen_grads_and_vars)

    #find the deltas
    #disc_delta = tf.nest.map_structure(tf.subtract, discriminator.weights,
                                        #from_server.discriminator_weights)

    #gen_delta = tf.nest.map_structure(tf.subtract, generator.weights,
                                        #from_server.generator_weights)
    # add to buffers
    #zero_disc = tf.nest.map_structure(lambda a, b: a + b, zero_disc, disc_delta)
    #zero_gen = tf.nest.map_structure(lambda a, b: a + b, zero_gen, gen_delta)

    num_examples += min_batch_size
  num_examples_float = tf.cast(num_examples, tf.float32)
  disc_delta = tf.nest.map_structure(lambda a, b: (a-b)/num_examples_float, zero_disc.weights, 
                  from_server.discriminator_weights)
  gen_delta = tf.nest.map_structure(lambda a,b: (a-b)/num_examples_float, zero_gen.weights,
                  from_server.generator_weights)

  disc_delta, disc_has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(disc_delta))
  gen_delta, gen_has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(gen_delta))   
  update_weight = tf.cast(num_examples, tf.float32)
  # Zero out the weight if there are any non-finite values.
  # TODO(b/122071074): federated_mean might not do the right thing if
  # all clients have zero weight.
  update_weight_disc = tf.cond(
      tf.equal(disc_has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))
  update_weight_gen = tf.cond(
      tf.equal(gen_has_non_finite_delta, 0), lambda: update_weight,
      lambda: tf.constant(0.0))   
  update_weight = tf.math.minimum(update_weight_disc, update_weight_gen) 
  return ClientOutput(
      discriminator_weights_delta=disc_delta,
      generator_weights_delta=gen_delta,
      update_weight=update_weight,
      counters={'num_discriminator_train_examples': num_examples})

def server_initial_state(generator, discriminator):
  """Returns the initial state of the server."""
  return ServerState(
      generator_weights=_weights(generator),
      discriminator_weights=_weights(discriminator),
      counters=collections.OrderedDict(
          num_discriminator_train_examples=tf.constant(0),
          num_generator_train_examples=tf.constant(0),
          num_rounds=tf.constant(0)),
      aggregation_state=())


@tf.function
def server_computation(
    # Tensor/Dataset arguments that will be supplied by TFF:
    server_state: ServerState,
    gen_delta: tf.keras.Model.weights,
    disc_delta: tf.keras.Model.weights,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer) -> ServerState:
  """The computation to run on the server, training the generator.

  Args:
    server_state: The initial `ServerState` for the round.
    gen_inputs_ds: An infinite `tf.data.Dataset` of inputs to the `generator`.
    client_output: The (possibly aggregated) `ClientOutput`.
    generator:  The generator.
    discriminator: The discriminator.
    server_disc_update_optimizer: Optimizer used to `apply_gradients` based on
      the client_output delta.
    train_generator_fn: A function which takes the two networks and generator
      input and trains the generator.
    new_aggregation_state: The updated state of the (possibly DP) averaging
      aggregator.

  Returns:
    An updated `ServerState` object.
  """
  # A tf.function can't modify the structure of its input arguments,
  # so we make a semi-shallow copy:
  server_state = attr.evolve(
      server_state, counters=collections.OrderedDict(server_state.counters))

  tf.nest.map_structure(lambda a, b: a.assign(b), generator.weights,
                        server_state.generator_weights)
  tf.nest.map_structure(lambda a, b: a.assign(b), discriminator.weights,
                        server_state.discriminator_weights)
  tf.nest.assert_same_structure(disc_delta, discriminator.weights)
  grads_and_vars_disc = tf.nest.map_structure(lambda x, v: (-1.0 * x, v), disc_delta,
                                         discriminator.weights)
  grads_and_vars_gen = tf.nest.map_structure(lambda x, v: (-1.0 * x, v), gen_delta,
                                         generator.weights)
  disc_optimizer.apply_gradients(
      grads_and_vars_disc, name='server_update_disc')
  gen_optimizer.apply_gradients(
      grads_and_vars_gen, name='server_update_gen'
  )

  gen_examples_this_round = tf.constant(0)
  
  numrounds = server_state.counters['num_rounds']
  round_mod = tf.math.floormod(numrounds, 2, name=None)
  server_state.meta_gen = tf.cond(tf.math.equal(round_mod,0), lambda: generator.weights, 
          lambda: server_state.generator_weights)
  server_state.meta_disc = tf.cond(tf.math.equal(round_mod,0), lambda: discriminator.weights, 
          lambda: server_state.discriminator_weights)
  
  server_state.counters[
      'num_generator_train_examples'] += gen_examples_this_round
  server_state.counters['num_rounds'] += 1
  server_state.generator_weights = _weights(generator)
  server_state.discriminator_weights = _weights(discriminator)
  return server_state


def create_train_generator_fn(gan_loss_fns: gan_losses.AbstractGanLossFns,
                              gen_optimizer: tf.keras.optimizers.Optimizer):
  """Create a function that trains generator, binding loss and optimizer.

  Args:
    gan_loss_fns: Instance of gan_losses.AbstractGanLossFns interface,
      specifying the generator/discriminator training losses.
    gen_optimizer: Optimizer for training the generator.

  Returns:
    Function that executes one step of generator training.
  """
  # We check that the optimizer has not been used previously, which ensures
  # that when it is bound the train fn isn't holding onto a different copy of
  # the optimizer variables then the copy that is being exchanged b/w server and
  # clients.
  if gen_optimizer.variables():
    raise ValueError(
        'Expected gen_optimizer to not have been used previously, but '
        'variables were already initialized.')

  @tf.function
  def train_generator_fn(generator: tf.keras.Model,
                         discriminator: tf.keras.Model, generator_inputs):
    """Trains the generator on a single batch.

    Args:
      generator:  The generator.
      discriminator: The discriminator.
      generator_inputs: A batch of inputs (usually noise) for the generator.

    Returns:
      The number of examples trained on.
    """

    def gen_loss():
      """Does the forward pass and computes losses for the generator."""
      # N.B. The complete pass must be inside loss() for gradient tracing.
      return gan_loss_fns.generator_loss(generator, discriminator,
                                         generator_inputs)

    gen_optimizer.minimize(gen_loss, var_list=generator.trainable_variables)
    return tf.shape(generator_inputs)[0]

  return train_generator_fn


def create_train_discriminator_fn(
    gan_loss_fns: gan_losses.AbstractGanLossFns,
    disc_optimizer: tf.keras.optimizers.Optimizer):
  """Create a function that trains discriminator, binding loss and optimizer.

  Args:
    gan_loss_fns: Instance of gan_losses.AbstractGanLossFns interface,
      specifying the generator/discriminator training losses.
    disc_optimizer: Optimizer for training the discriminator.

  Returns:
    Function that executes one step of discriminator training.
  """
  # We assert that the optimizer has not been used previously, which ensures
  # that when it is bound the train fn isn't holding onto a different copy of
  # the optimizer variables then the copy that is being exchanged b/w server and
  # clients.
  if disc_optimizer.variables():
    raise ValueError(
        'Expected disc_optimizer to not have been used previously, but '
        'variables were already initialized.')

  @tf.function
  def train_discriminator_fn(generator: tf.keras.Model,
                             discriminator: tf.keras.Model, generator_inputs,
                             real_data):
    """Trains the discriminator on a single batch.

    Args:
      generator:  The generator.
      discriminator: The discriminator.
      generator_inputs: A batch of inputs (usually noise) for the generator.
      real_data: A batch of real data for the discriminator.

    Returns:
      The size of the batch.
    """

    def disc_loss():
      """Does the forward pass and computes losses for the discriminator."""
      # N.B. The complete pass must be inside loss() for gradient tracing.
      return gan_loss_fns.discriminator_loss(generator, discriminator,
                                             generator_inputs, real_data)

    disc_optimizer.minimize(
        disc_loss, var_list=discriminator.trainable_variables)
    return tf.shape(real_data)[0]

  return train_discriminator_fn
