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
"""Tests for ResNet v2 models."""

from absl.testing import parameterized
import tensorflow as tf

from generalization.utils import resnet_models


class ResnetModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('resnet18', resnet_models.create_resnet18),
      ('resnet34', resnet_models.create_resnet34),
      ('resnet50', resnet_models.create_resnet50),
      ('resnet152', resnet_models.create_resnet152),
  )
  def test_resnet_constructs_with_imagenet_inputs(self, resnet_constructor):
    model = resnet_constructor(input_shape=(224, 224, 3), num_classes=1000)
    self.assertIsInstance(model, tf.keras.Model)

  def test_bad_input_raises_exception(self):
    with self.assertRaises(Exception):
      resnet_models.create_resnet50(input_shape=(1, 1), num_classes=10)

  def test_batch_norm_constructs(self):
    batch_resnet = resnet_models.create_resnet(
        input_shape=(32, 32, 3), num_classes=10, norm='batch')
    self.assertIsInstance(batch_resnet, tf.keras.Model)

  def test_group_norm_constructs(self):
    group_resnet = resnet_models.create_resnet(
        input_shape=(32, 32, 3), num_classes=10, norm='group')
    self.assertIsInstance(group_resnet, tf.keras.Model)

  def test_basic_fewer_parameters_than_bottleneck(self):
    input_shape = (32, 32, 3)
    num_classes = 10
    basic_resnet = resnet_models.create_resnet(
        input_shape, num_classes, block='basic')
    bottleneck_resnet = resnet_models.create_resnet(
        input_shape, num_classes, block='bottleneck')

    self.assertLess(basic_resnet.count_params(),
                    bottleneck_resnet.count_params())

  def test_repetitions_increase_number_parameters(self):
    input_shape = (32, 32, 3)
    num_classes = 10
    small_resnet = resnet_models.create_resnet(
        input_shape, num_classes, repetitions=[1, 1])
    big_resnet = resnet_models.create_resnet(
        input_shape, num_classes, repetitions=[2, 2])
    self.assertLess(small_resnet.count_params(), big_resnet.count_params())

  @parameterized.named_parameters(
      ('resnet18', resnet_models.create_resnet18),
      ('resnet34', resnet_models.create_resnet34),
      ('resnet50', resnet_models.create_resnet50),
      ('resnet152', resnet_models.create_resnet152),
  )
  def test_model_initialization_uses_random_seed(self, resnet_constructor):
    model_1_with_seed_0 = resnet_constructor(
        input_shape=(32, 32, 3), num_classes=100, seed=0)
    model_2_with_seed_0 = resnet_constructor(
        input_shape=(32, 32, 3), num_classes=100, seed=0)
    model_1_with_seed_1 = resnet_constructor(
        input_shape=(32, 32, 3), num_classes=100, seed=1)
    model_2_with_seed_1 = resnet_constructor(
        input_shape=(32, 32, 3), num_classes=100, seed=1)
    self.assertAllClose(model_1_with_seed_0.weights,
                        model_2_with_seed_0.weights)
    self.assertAllClose(model_1_with_seed_1.weights,
                        model_2_with_seed_1.weights)
    self.assertNotAllClose(model_1_with_seed_0.weights,
                           model_1_with_seed_1.weights)


if __name__ == '__main__':
  tf.test.main()
