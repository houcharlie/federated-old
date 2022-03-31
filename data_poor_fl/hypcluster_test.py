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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import hypcluster


def create_dataset():
  # Create data satisfying y = x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[2.0], [3.0], [4.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def get_input_spec():
  return create_dataset().element_spec


def model_fn(initializer='zeros'):
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer=initializer,
          bias_initializer=initializer,
          input_shape=(1,))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


@tff.tf_computation
def create_nested_structure():
  return [
      dict(a=tf.zeros((2, 2), dtype=tf.int32), b=1, c=3),
      dict(a=tf.ones((2, 2), dtype=tf.int32), b=2, c=4),
      dict(a=2 * tf.ones((2, 2), dtype=tf.int32), b=3, c=5),
  ]


def create_initial_models(num_models: int):
  model = model_fn(initializer='ones')
  model_weights_tensors = tf.nest.map_structure(
      lambda var: var.numpy(), tff.learning.ModelWeights.from_model(model))
  return [model_weights_tensors for _ in range(num_models)]


class GatherTest(tf.test.TestCase):

  def test_gather_type_signature(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    gather_fn = hypcluster.build_gather_fn(list_element_type, num_indices=3)
    self.assertEqual(gather_fn.type_signature.parameter[0], list_type)
    self.assertEqual(gather_fn.type_signature.parameter[1],
                     tff.TensorType(dtype=tf.int32))
    self.assertEqual(gather_fn.type_signature.result, list_element_type)

  def test_gather_on_list_of_tensors(self):
    list_element_type = tff.TensorType(dtype=tf.int32)
    gather_fn = hypcluster.build_gather_fn(list_element_type, num_indices=5)
    gather_structure = [5, 1, 16, -1, 10042]
    for i in range(5):
      actual_result = gather_fn(gather_structure, i)
      expected_result = gather_structure[i]
      self.assertAllEqual(actual_result, expected_result)

  def test_gather_on_nested_structure(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    gather_fn = hypcluster.build_gather_fn(list_element_type, num_indices=3)
    gather_structure = create_nested_structure()
    for i in range(3):
      actual_result = gather_fn(gather_structure, i)
      expected_result = gather_structure[i]
      self.assertDictEqual(actual_result, expected_result)


class ScatterTest(tf.test.TestCase):

  def test_scatter_type_signature(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    scatter_fn = hypcluster.build_scatter_fn(list_element_type, num_indices=3)
    self.assertEqual(scatter_fn.type_signature.parameter[0], list_element_type)
    self.assertEqual(scatter_fn.type_signature.parameter[1],
                     tff.TensorType(tf.int32))
    self.assertEqual(scatter_fn.type_signature.parameter[2],
                     tff.TensorType(tf.float32))
    self.assertEqual(scatter_fn.type_signature.result[0], list_type)
    expected_result_weight_type = tff.StructWithPythonType(
        [tff.TensorType(tf.float32)] * 3, list)
    self.assertEqual(scatter_fn.type_signature.result[1],
                     expected_result_weight_type)

  def test_scatter_tensor(self):
    value_type = tff.TensorType(dtype=tf.int32)
    scatter_fn = hypcluster.build_scatter_fn(value_type, num_indices=5)
    for i in range(5):
      actual_value, actual_weight = scatter_fn(7, i, 0.5)
      expected_value = [0] * 5
      expected_value[i] = 7
      self.assertEqual(actual_value, expected_value)
      expected_weight = [0.0] * 5
      expected_weight[i] = 0.5
      self.assertEqual(actual_weight, expected_weight)

  def test_nested_structure(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    nested_structure = create_nested_structure()
    scatter_fn = hypcluster.build_scatter_fn(list_element_type, num_indices=2)
    actual_value, actual_weight = scatter_fn(nested_structure[1], 0, 3.0)
    expected_value = [
        nested_structure[1],
        dict(a=tf.zeros((2, 2), dtype=tf.int32), b=0, c=0)
    ]
    for actual_dict, expected_dict in zip(actual_value, expected_value):
      self.assertDictEqual(actual_dict, expected_dict)
    # self.assertAllEqual(expected_value, actual_value)
    expected_weight = [3.0, 0.0]
    self.assertEqual(actual_weight, expected_weight)


class HypClusterTrainTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('clusters1_with_init', 1, create_initial_models(1)),
      ('clusters1_without_init', 1, None),
      ('clusters2_without_int', 2, None),
      ('clusters3_with_init', 3, create_initial_models(3)),
      ('clusters5_without_init', 5, None),
  )
  def test_constructs_with_default_aggregator(self, num_clusters,
                                              initial_model_weights_list):
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    hyp_alg = hypcluster.build_hypcluster_train(
        model_fn=model_fn,
        num_clusters=num_clusters,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        initial_model_weights_list=initial_model_weights_list)
    state = hyp_alg.initialize()
    self.assertLen(state.global_model_weights, num_clusters)
    self.assertLen(state.aggregator, num_clusters)
    self.assertLen(state.finalizer, num_clusters)
    if initial_model_weights_list:
      tf.nest.map_structure(self.assertAllEqual, state.global_model_weights,
                            initial_model_weights_list)

  @parameterized.named_parameters(
      ('clusters1_with_init', 1, create_initial_models(1)),
      ('clusters1_without_init', 1, None),
      ('clusters2_without_int', 2, None),
      ('clusters3_with_init', 3, create_initial_models(3)),
      ('clusters5_without_init', 5, None),
  )
  def test_constructs_with_non_default_aggregator(self, num_clusters,
                                                  initial_model_weights_list):
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    hyp_alg = hypcluster.build_hypcluster_train(
        model_fn=model_fn,
        num_clusters=num_clusters,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        model_aggregator=tff.learning.robust_aggregator(),
        initial_model_weights_list=initial_model_weights_list)
    state = hyp_alg.initialize()
    self.assertLen(state.global_model_weights, num_clusters)
    self.assertLen(state.aggregator, num_clusters)
    self.assertLen(state.finalizer, num_clusters)
    if initial_model_weights_list:
      tf.nest.map_structure(self.assertAllEqual, state.global_model_weights,
                            initial_model_weights_list)

  def test_construction_fails_with_mismatched_initial_models(self):
    num_clusters = 1
    initial_model_weights_list = create_initial_models(2)
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    with self.assertRaisesRegex(ValueError, 'does not equal'):
      hypcluster.build_hypcluster_train(
          model_fn=model_fn,
          num_clusters=num_clusters,
          client_optimizer=client_optimizer,
          server_optimizer=server_optimizer,
          initial_model_weights_list=initial_model_weights_list)

  def test_matches_fed_avg_with_one_cluster(self):
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    fed_avg = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer,
        server_optimizer_fn=server_optimizer)
    hyp_alg = hypcluster.build_hypcluster_train(
        model_fn=model_fn,
        num_clusters=1,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer)

    fed_avg_state = fed_avg.initialize()
    hyp_alg_state = hyp_alg.initialize()
    self.assertAllClose(
        fed_avg.get_model_weights(fed_avg_state).trainable,
        hyp_alg.get_model_weights(hyp_alg_state)[0].trainable)
    federated_data = [create_dataset(), create_dataset()]
    for _ in range(5):
      fed_avg_output = fed_avg.next(fed_avg_state, federated_data)
      fed_avg_state = fed_avg_output.state
      hyp_alg_output = hyp_alg.next(hyp_alg_state, federated_data)
      hyp_alg_state = hyp_alg_output.state
      self.assertAllClose(
          fed_avg.get_model_weights(fed_avg_state).trainable,
          hyp_alg.get_model_weights(hyp_alg_state)[0].trainable)


class HypClusterEvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('clusters1', 1),
      ('clusters2', 2),
      ('clusters3', 3),
      ('clusters5', 5),
  )
  def test_constructs(self, num_clusters):
    hyp_eval = hypcluster.build_hypcluster_eval(
        model_fn=model_fn, num_clusters=num_clusters)
    self.assertLen(hyp_eval.type_signature.parameter[0].member, num_clusters)

  def test_matches_federated_eval_with_one_cluster(self):
    hyp_eval = hypcluster.build_hypcluster_eval(
        model_fn=model_fn, num_clusters=1)
    federated_eval = tff.learning.build_federated_evaluation(model_fn)
    model_weights = tff.learning.ModelWeights.from_model(model_fn())
    federated_data = [create_dataset(), create_dataset()]
    hyp_metrics = hyp_eval([model_weights], federated_data)
    reference_metrics = federated_eval(model_weights, federated_data)
    self.assertAllClose(hyp_metrics, reference_metrics)

  def test_selects_best_model(self):
    hyp_eval = hypcluster.build_hypcluster_eval(
        model_fn=model_fn, num_clusters=2)
    zero_model = model_fn(initializer='zeros')
    ones_model = model_fn(initializer='ones')
    zero_weights = tff.learning.ModelWeights.from_model(zero_model)
    ones_weights = tff.learning.ModelWeights.from_model(ones_model)
    federated_data = [create_dataset()]
    hyp_metrics = hyp_eval([zero_weights, ones_weights], federated_data)
    federated_eval = tff.learning.build_federated_evaluation(model_fn)
    reference_metrics = federated_eval(ones_weights, federated_data)
    self.assertAllClose(hyp_metrics, reference_metrics)


if __name__ == '__main__':
  tf.test.main()
