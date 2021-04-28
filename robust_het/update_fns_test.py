import tensorflow as tf 
import collections
import update_fns 
import federated_trainer
import process_builder
import tensorflow_federated as tff
import numpy as np
def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)
def new_element():
    return collections.OrderedDict(
        {'x': tf.convert_to_tensor(np.random.randint(255, size = (5,784)), dtype = tf.uint8),
        'y': tf.convert_to_tensor(np.random.randint(10, size = (5,1)), dtype=tf.int64)})
class UpdateFnsTest(tf.test.TestCase):
    def test_project_to_simplex(self):
        p = tf.random.uniform(shape=[10])
        p_ = update_fns.project_to_simplex(p)
        self.assertAllClose(self.evaluate(tf.reduce_sum(p_)), 1, 1e-7)
    
    def test_server_update_fn(self):
        model = federated_trainer.model_fn()
        meta_update_interval = 3
        server_optimizer = tf.keras.optimizers.SGD(learning_rate = 1.)
        w = _get_weights(model)
        meta_w = _get_weights(federated_trainer.model_fn())
        server_state = process_builder.ServerState(
            w = w,
            meta_w = meta_w,
            round_num = tf.constant(0.0)
        )
        weights_delta = tf.nest.map_structure(
            tf.ones_like, _get_weights(model).trainable)
        server_state = update_fns.server_update(model, meta_update_interval, 
                                server_optimizer, server_state,
                                weights_delta)
        tf.nest.map_structure(lambda x, y: self.assertAllClose(x,y,1e-7),
                self.evaluate(server_state.w),
                self.evaluate(server_state.meta_w))

    def test_client_update_fn(self):
        
        dataset1 = [new_element()]
        model = federated_trainer.model_fn()
        
        meta_update_interval = 5
        w = tf.nest.map_structure(tf.ones_like,_get_weights(model))
        meta_w = tf.nest.map_structure(tf.ones_like,_get_weights(model))
        server_state = process_builder.ServerState(
            w = w,
            meta_w = meta_w,
            round_num = tf.constant(1.0)
        )
        client_lr = 1.
        p_size = 5
        p = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        p_even = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        client_optimizer = tf.keras.optimizers.SGD(learning_rate = client_lr)
        tau = 0
        shift_reg = 0
        reg = 0 
        w_correction = tf.nest.map_structure(
            tf.zeros_like, _get_weights(model).trainable)
        client_state = process_builder.ClientState(
            p = tf.Variable(tf.fill(p_size, 1./float(p_size))),
            meta_p = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        )
        from_server = process_builder.FromServer(
            w = server_state.w,
            meta_w = server_state.meta_w,
            round_num = server_state.round_num
        )
        ### CHECK SCAFFOLD CORRECTION
        client_output = update_fns.client_update(model, p, p_even, 
                    client_optimizer, tau, p_size, shift_reg,
                    reg, meta_update_interval, client_lr, dataset1, client_state,
                    from_server, w_correction)
        without_correction = self.evaluate(client_output.w_delta)
        
        w_correction = tf.nest.map_structure(
            tf.ones_like, _get_weights(model).trainable)
        client_output = update_fns.client_update(model, p, p_even, 
                    client_optimizer, tau, p_size, shift_reg,
                    reg, meta_update_interval, client_lr, dataset1, client_state,
                    from_server, w_correction)
        with_correction = self.evaluate(client_output.w_delta) 
        
        tf.nest.map_structure(
            lambda x, y: self.assertAllClose(x - 1, y, 1e-7),
            without_correction, with_correction
        )
    def test_client_update_catalyst(self):
        ### CHECK CATALYST
        dataset1 = [new_element(), new_element(), new_element()]
        model = federated_trainer.model_fn()
        
        meta_update_interval = 5
        w = tf.nest.map_structure(tf.ones_like,_get_weights(model))
        meta_w = tf.nest.map_structure(tf.ones_like,_get_weights(model))
        server_state = process_builder.ServerState(
            w = w,
            meta_w = meta_w,
            round_num = tf.constant(1.0)
        )
        client_lr = 0.01
        p_size = 5
        p = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        p_even = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        client_optimizer = tf.keras.optimizers.SGD(learning_rate = client_lr)
        tau = 0.
        shift_reg = 0
        reg = 0 
        w_correction = tf.nest.map_structure(
            tf.zeros_like, _get_weights(model).trainable)
        client_state = process_builder.ClientState(
            p = tf.Variable(tf.fill(p_size, 1./float(p_size))),
            meta_p = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        )
        from_server = process_builder.FromServer(
            w = server_state.w,
            meta_w = server_state.meta_w,
            round_num = server_state.round_num
        )
        client_output = update_fns.client_update(model, p, p_even, 
                    client_optimizer, tau, p_size, shift_reg,
                    reg, meta_update_interval, client_lr, dataset1, client_state,
                    from_server, w_correction)
        without_correction = self.evaluate(client_output.w_delta)
        p_without_correction = self.evaluate(
            tf.nn.l2_loss(
                client_output.client_state.p - client_output.client_state.meta_p)
        )
        tau = 2.
        client_output = update_fns.client_update(model, p, p_even, 
                    client_optimizer, tau, p_size, shift_reg,
                    reg, meta_update_interval, client_lr, dataset1, client_state,
                    from_server, w_correction)
        with_tau = self.evaluate(client_output.w_delta)
        p_tau = self.evaluate(
            tf.nn.l2_loss(
                client_output.client_state.p - client_output.client_state.meta_p)
        )
        
        self.assertLess(
                p_tau, p_without_correction
            )
        tf.nest.map_structure(
            lambda x, y: self.assertLess(
                tf.nn.l2_loss(x), tf.nn.l2_loss(y)
            ),
            with_tau, without_correction
        )

        



        
    



if __name__ == '__main__':
    tf.test.main()