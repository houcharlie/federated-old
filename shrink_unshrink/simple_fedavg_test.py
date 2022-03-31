# Copyright 2020, The TensorFlow Federated Authors.
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
"""End-to-end example testing Federated Averaging with CNN and RNN."""

import collections
import functools
from typing import Callable, List, OrderedDict
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink import simple_fedavg_tf
from shrink_unshrink import simple_fedavg_tff


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def _create_random_batch():
  return collections.OrderedDict(
      x=tf.random.uniform(tf.TensorShape([1, 28, 28, 1]), dtype=tf.float32),
      y=tf.constant(1, dtype=tf.int32, shape=[1]))


def _simple_fedavg_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return simple_fedavg_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


def _tff_learning_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32))
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def _create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def _mnist_inference(variables, inputs):
  logits = tf.nn.softmax(tf.matmul(inputs, variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
  return logits, predictions


def _mnist_forward_pass(variables, batch):
  y, predictions = _mnist_inference(variables, batch['x'])

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return tff.learning.BatchOutput(
      loss=loss, predictions=predictions, num_examples=num_examples)


class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = _create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def weights(self):
    return tff.learning.ModelWeights(
        trainable=self.trainable_variables,
        non_trainable=self.non_trainable_variables)

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32))

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training  # Unused.
    return _mnist_inference(self._variables, batch)

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    return _mnist_forward_pass(self._variables, batch)

  @tf.function
  def report_local_unfinalized_metrics(
      self) -> OrderedDict[str, List[tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to unfinalized values."""
    return collections.OrderedDict(
        num_examples=[self._variables.num_examples],
        loss=[self._variables.loss_sum, self._variables.num_examples],
        accuracy=[self._variables.accuracy_sum, self._variables.num_examples])

  def metric_finalizers(
      self) -> OrderedDict[str, Callable[[List[tf.Tensor]], tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to finalizers."""
    return collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x[0]),
        loss=tf.function(func=lambda x: x[0] / x[1]),
        accuracy=tf.function(func=lambda x: x[0] / x[1]))

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    raise NotImplementedError(
        'The `reset_metrics` method isn\'t implemented for your custom '
        '`tff.learning.Model`. Please implement it before using this method. '
        'You can leave this method unimplemented if you won\'t use this method.'
    )


def _create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[[5]], pixels=[np.random.rand(28, 28).astype(np.float32)])
  dataset = tf.data.Dataset.from_tensor_slices(emnist_batch)

  def client_data():
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2).batch(2)

  return client_data


class SimpleFedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('tff.learning.Model_wrapper', _tff_learning_model_fn))
  def test_data_type_signature(self, model_fn):
    it_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=model_fn,
        client_model_fn=model_fn,
        shrink_unshrink_info=make_whimsty_shrink_unshrink_info())
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type),
        '{<x=float32[?,28,28,1],y=int32[?]>*}@CLIENTS')

  @parameterized.named_parameters(
      ('tff.learning.Model_wrapper', _tff_learning_model_fn))
  def test_simple_training(self, model_fn):
    it_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=model_fn,
        client_model_fn=model_fn,
        shrink_unshrink_info=make_whimsty_shrink_unshrink_info())
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for _ in range(3):
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss['loss'])

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_self_contained_example_custom_model(self):

    client_data = _create_client_data()
    train_data = [client_data()]

    trainer = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=MnistModel,
        client_model_fn=MnistModel,
        shrink_unshrink_info=make_whimsty_shrink_unshrink_info())
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, loss = trainer.next(state, train_data)
      losses.append(loss['loss'])
    self.assertLess(losses[1], losses[0])

  def test_keras_evaluate(self):
    keras_model = _create_test_cnn_model()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy = simple_fedavg_tf.keras_evaluate(keras_model, sample_data, metric)
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)

  def test_tff_learning_evaluate(self):
    it_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=_tff_learning_model_fn,
        client_model_fn=_tff_learning_model_fn,
        shrink_unshrink_info=make_whimsty_shrink_unshrink_info())
    server_state = it_process.initialize()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    keras_model = _create_test_cnn_model()
    server_state.model_weights.assign_weights_to(keras_model)

    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32))
    ]
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy = simple_fedavg_tf.keras_evaluate(keras_model, sample_data, metric)
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)


def _server_init(model, optimizer):
  """Returns initial `ServerState`.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A `ServerState` namedtuple.
  """
  simple_fedavg_tff._initialize_optimizer_vars(model, optimizer)
  return simple_fedavg_tf.ServerState(
      model_weights=model.weights,
      optimizer_state=optimizer.variables(),
      round_num=0,
      shrink_unshrink_server_info=simple_fedavg_tf.ShrinkUnshrinkServerInfo(
          lmbda=0.0, oja_left_maskval_to_projmat_dict=[], memory_dict=[]))


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    model = model_fn()
    optimizer = optimizer_fn()
    state = _server_init(model, optimizer)
    weights_delta = tf.nest.map_structure(tf.ones_like,
                                          model.trainable_variables)

    for _ in range(2):
      state = simple_fedavg_tf.server_update(model, optimizer, state,
                                             weights_delta)

    model_vars = self.evaluate(state.model_weights)
    train_vars = model_vars.trainable
    self.assertLen(train_vars, 2)
    self.assertEqual(state.round_num, 2)
    # weights are initialized with all-zeros, weights_delta is all ones,
    # SGD learning rate is 0.1. Updating server for 2 steps.
    self.assertAllClose(train_vars, [np.ones_like(v) * 0.2 for v in train_vars])

  def test_self_contained_example_custom_model(self):
    self._assert_server_update_with_all_ones(MnistModel)


class ClientTest(tf.test.TestCase):

  def test_self_contained_example(self):

    client_data = _create_client_data()

    model = MnistModel()
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    losses = []
    for r in range(2):
      optimizer = optimizer_fn()
      simple_fedavg_tff._initialize_optimizer_vars(model, optimizer)
      server_message = simple_fedavg_tf.BroadcastMessage(
          model_weights=model.weights, round_num=r)
      outputs = simple_fedavg_tf.client_update(model, client_data(),
                                               server_message, optimizer)
      loss_finalizer = model.metric_finalizers()['loss']
      losses.append(loss_finalizer(outputs.model_output['loss']))

    self.assertAllEqual(int(outputs.client_weight.numpy()), 2)
    self.assertLess(losses[1], losses[0])


def _create_test_rnn_model(vocab_size: int = 6,
                           sequence_length: int = 5,
                           mask_zero: bool = True) -> tf.keras.Model:
  """A simple RNN model for test."""
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size,
          input_length=sequence_length,
          output_dim=8,
          mask_zero=mask_zero))
  model.add(
      tf.keras.layers.LSTM(
          units=16,
          kernel_initializer='he_normal',
          return_sequences=True,
          stateful=False))
  model.add(tf.keras.layers.Dense(vocab_size))
  return model


def _rnn_model_fn() -> tff.learning.Model:
  keras_model = _create_test_rnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 5], tf.int32),
      y=tf.TensorSpec([None, 5], tf.int32))
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


def make_whimsty_shrink_unshrink_info():
  return simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
      left_mask=[0],
      right_mask=[0],
      build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix,
      new_projection_dict_decimate=1)


class RNNTest(tf.test.TestCase, parameterized.TestCase):

  def test_build_fedavg_process(self):
    it_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=_rnn_model_fn,
        client_model_fn=_rnn_model_fn,
        shrink_unshrink_info=make_whimsty_shrink_unshrink_info())
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_type = it_process.next.type_signature.parameter
    model_type = tff.learning.framework.weights_type_from_model(_rnn_model_fn)
    self.assertEqual(
        str(federated_type[0]),
        '<model_weights=<trainable=<float32[6,8],float32[8,64],float32[16,64],float32[64],float32[16,6],float32[6]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<0=float32[?,?]>,memory_dict=<0=float32[?,?]>>>@SERVER'  # pylint:disable=too-many-format-args
        .format(model_type))
    self.assertEqual(
        str(federated_type[1]), '{<x=int32[?,5],y=int32[?,5]>*}@CLIENTS')

  def test_client_adagrad_train(self):
    it_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=_rnn_model_fn,
        client_model_fn=_rnn_model_fn,
        client_optimizer_fn=functools.partial(
            tf.keras.optimizers.Adagrad, learning_rate=0.01),
        shrink_unshrink_info=make_whimsty_shrink_unshrink_info())
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for _ in range(3):
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss['loss'])

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])


if __name__ == '__main__':
  tf.test.main()
