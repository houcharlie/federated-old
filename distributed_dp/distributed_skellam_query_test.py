# Copyright 2018, The TensorFlow Authors.
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
"""Basic tests for DistributedSkellamSumQuery."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import test_utils
import tensorflow_probability as tf_prob

from distributed_dp import distributed_skellam_query


class DistributedSkellamQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_skellam_sum_no_noise(self):
    with self.cached_session() as sess:
      record1 = tf.constant([2, 0], dtype=tf.int32)
      record2 = tf.constant([-1, 1], dtype=tf.int32)

      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=10, l2_norm_bound=10, local_stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1, 1]
      self.assertAllClose(result, expected, atol=0)

  def test_skellam_multiple_shapes(self):
    with self.cached_session() as sess:
      tensor1 = tf.constant([2, 0], dtype=tf.int32)
      tensor2 = tf.constant([-1, 1, 3], dtype=tf.int32)
      record = [tensor1, tensor2]

      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=10, l2_norm_bound=10, local_stddev=0.0)
      query_result, _ = test_utils.run_query(query, [record, record])
      result = sess.run(query_result)
      expected = [2 * tensor1, 2 * tensor2]
      self.assertAllClose(result, expected, atol=0)

  def test_skellam_raise_type_exception(self):
    record1 = tf.constant([2, 0], dtype=tf.float32)
    record2 = tf.constant([-1, 1], dtype=tf.float32)
    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=10, l2_norm_bound=10, local_stddev=0.0)

    with self.cached_session() as sess, self.assertRaises(TypeError):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      sess.run(query_result)

  def test_skellam_raise_l1_norm_exception(self):
    record1 = tf.constant([1, 2], dtype=tf.int32)
    record2 = tf.constant([3, 4], dtype=tf.int32)
    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=1, l2_norm_bound=100, local_stddev=0.0)

    with self.cached_session() as sess, self.assertRaises(
        tf.errors.InvalidArgumentError):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      sess.run(query_result)

  def test_skellam_raise_l2_norm_exception(self):
    record1 = tf.constant([1, 2], dtype=tf.int32)
    record2 = tf.constant([3, 4], dtype=tf.int32)
    query = distributed_skellam_query.DistributedSkellamSumQuery(
        l1_norm_bound=10, l2_norm_bound=4, local_stddev=0.0)

    with self.cached_session() as sess, self.assertRaises(
        tf.errors.InvalidArgumentError):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      sess.run(query_result)

  def test_skellam_sum_with_noise(self):
    """Use only one record to test std."""
    with self.cached_session() as sess:
      num_trials = 1000
      record = tf.constant([1] * num_trials, dtype=tf.int32)
      stddev = 1.0

      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=1000.0, l2_norm_bound=1000, local_stddev=stddev)
      query_result, _ = test_utils.run_query(query, [record])

      result_stddev = np.std(sess.run(query_result))
      self.assertNear(result_stddev, stddev, 0.1)

  def test_compare_centralized_distributed_skellam(self):
    """Compare the percentiles of distributed and centralized Skellam.

    The test creates a large zero-vector with shape [num_trials, num_users] to
    be processed with the distributed Skellam noise stddev=1. The result is
    summed over the num_users dimension. The centralized result is produced by
    adding noise to a zero vector [num_trials] with stddev = 1*sqrt(num_users).
    Both results are evaluated to match percentiles (25, 50, 75).
    """

    with self.cached_session() as sess:
      num_trials = 10000
      num_users = 100
      record = tf.zeros([num_trials], dtype=tf.int32)
      local_stddev = 1.0
      query = distributed_skellam_query.DistributedSkellamSumQuery(
          l1_norm_bound=10.0, l2_norm_bound=10, local_stddev=local_stddev)

      query_result, _ = test_utils.run_query(query, [record] * num_users)
      distributed_noised = self.evaluate(query_result)

      def add_noise(v, stddev):
        lam = stddev**2 / 2
        noise_poisson1 = tf.random.poisson(
            lam=lam, shape=tf.shape(v), dtype=v.dtype)
        noise_poisson2 = tf.random.poisson(
            lam=lam, shape=tf.shape(v), dtype=v.dtype)
        res = v + (noise_poisson1 - noise_poisson2)
        return res

      record_centralized = tf.zeros([num_trials], dtype=tf.int32)
      centralized_noised = sess.run(
          add_noise(record_centralized, local_stddev * np.sqrt(num_users)))

      tolerance = 5
      self.assertAllClose(
          tf_prob.stats.percentile(distributed_noised, 50.0),
          tf_prob.stats.percentile(centralized_noised, 50.0),
          atol=tolerance)
      self.assertAllClose(
          tf_prob.stats.percentile(distributed_noised, 75.0),
          tf_prob.stats.percentile(centralized_noised, 75.0),
          atol=tolerance)
      self.assertAllClose(
          tf_prob.stats.percentile(distributed_noised, 25.0),
          tf_prob.stats.percentile(centralized_noised, 25.0),
          atol=tolerance)


if __name__ == '__main__':
  tf.test.main()
