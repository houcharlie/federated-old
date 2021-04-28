import collections
from typing import Callable, Optional, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[], tf.keras.optimizers.Optimizer]

@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
    """Structure for state on the server.
    Fields:
    -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
    -   `optimizer_state`: The server optimizer variables.
    -   `round_num`: The current training round, as a float.
    """
    w = attr.ib()
    meta_w = attr.ib()
    round_num = attr.ib()
    # This is a float to avoid type incompatibility when calculating learning 
    # rate
    # schedules.

@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
    """Structure for outputs returned from clients during federated optimization.
    Fields:
    -   `weights_delta`: A dictionary of updates to the model's trainable
        variables.
    -   `client_weight`: Weight to be used in a weighted mean when
        aggregating `weights_delta`.
    -   `model_output`: A structure matching
        `tff.learning.Model.report_local_outputs`, reflecting the results of
        training on the input dataset.
    -   `optimizer_output`: Additional metrics or other outputs defined by the
        optimizer.
    """
    w_delta = attr.ib()
    client_weight = attr.ib()
    client_state = attr.ib()
    model_output = attr.ib()
    optimizer_output = attr.ib()

@attr.s(eq=False, order=False, frozen=True)
class ClientState(object):
    """Structure for outputs returned from clients during federated optimization.
    Fields:
    -   `weights_delta`: A dictionary of updates to the model's trainable
        variables.
    -   `client_weight`: Weight to be used in a weighted mean when
        aggregating `weights_delta`.
    -   `model_output`: A structure matching
        `tff.learning.Model.report_local_outputs`, reflecting the results of
        training on the input dataset.
    -   `optimizer_output`: Additional metrics or other outputs defined by the
        optimizer.
    """
    meta_p = attr.ib()
    p = attr.ib()

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
    w = attr.ib()
    meta_w = attr.ib()
    round_num = attr.ib()

def build_server_init_fn(model_fn: ModelBuilder):
    @tff.tf_computation
    def init_tf():
        model = model_fn()
        server_state = ServerState(
            w=_get_weights(model),
            meta_w = _get_weights(model_fn()),
            round_num=1.0)
        return server_state
    return server_init_tf

def build_server_computation(meta_update_interval, model_fn, server_lr,
                            server_state_type, 
                            delta_type):
    @tff.tf_computation(server_state_type, delta_type)
    def server_update_fn(server_state, w_delta):
        return server_update(model_fn(), 
                            meta_update_interval, 
                            tf.keras.optimizers.SGD(learning_rate=server_lr),
                            server_state, 
                            w_delta)

def build_client_computation(model_fn,
                                p_size,
                                client_lr,
                                tau,
                                shift_reg,
                                reg,
                                meta_update_interval,
                                tf_dataset_type,
                                client_state_type, 
                                from_server_type, 
                                correction_type):
    @tff.tf_computation(tf_dataset_type,client_state_type, from_server_type, 
                        correction_type)
    def client_update_fn(tf_dataset, client_state, from_server_type, 
                        correction_type):
        p = tf.fill(p_size, 1./float(p_size))
        p_even = tf.fill(p_size, 1./float(p_size))
        return client_update(model_fn(), p, p_even,
                            tf.keras.optimizers.SGD(learning_rate=client_lr),
                            tau,
                            p_size,
                            shift_reg,
                            reg,
                            meta_update_interval,
                            client_lr,
                            tf_dataset, 
                            client_state,
                            from_server, 
                            correction)

def build_training_process(
        model_fn: ModelBuilder,
        server_lr: float,
        client_lr: float,
        tau:float,
        control:bool,
        meta_update_interval:int,
        p_size:int,
        shift_reg: float,
        reg: float):
    server_init_tf = build_server_init_fn(model_fn)
    dummy_model = model_fn()
    tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

    @tff.tf_computation(tf_dataset_type)
    def init_client_state(client_ds):
        p = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        meta_p = tf.Variable(tf.fill(p_size, 1./float(p_size)))
        return ClientState(p=p, meta_p=meta_p)

    server_state_type = server_init_tf.type_signature.result
    client_state_type = init_client_state.type_signature.result
    
    from_server_type = FromServer(
        w = server_state_type.w,
        meta_w = server_state_type.meta_w,
    )
    # last three are the types of:
    # dataset, client_state, from_server, and SCAFFOLD correction
    client_computation = build_client_computation(
                                model_fn,
                                p_size,
                                client_lr,
                                tau,
                                shift_reg,
                                reg,
                                meta_update_interval,
                                tf_dataset_type,
                                client_state_type, 
                                from_server_type, 
                                server_state_type.w)
    # last two are the types of:
    # server_state and w delta
    server_computation = build_server_computation(meta_update_interval,
                                model_fn, 
                                server_lr,
                                server_state_type, server_state_type.w)
    # last two are the types of:
    # dataset, client_state, from_server
    control_computation = build_client_computation(
                                model_fn,
                                p_size,
                                client_lr,
                                tau,
                                shift_reg,
                                reg,
                                meta_update_interval,
                                tf_dataset_type,
                                client_state_type, 
                                from_server_type, 
                                server_state_type.w)

    
    
    client_output_type = client_computation.type_signature.result

    @tff.federated_computation(tff.type_at_clients(tf_dataset_type))
    def initialize_fn(federated_dataset):
        server_state = tff.federated_value(server_init_tf(), tff.SERVER)
        client_states = tff.federated_map(init_client_state, 
                            (federated_dataset))
        return (server_state, client_states)
    
    @tff.federated_computation(
      tff.type_at_server(server_state_type),
      tff.type_at_clients(client_state_type),
      tff.type_at_clients(tf_dataset_type))
    def run_one_round(server_state, client_states, federated_dataset,
                        federated_dataset_single):
        from_server = FromServer(w = server_state.w,
                                    meta_w = server_state.meta_w,
                                    round_num = server_state.round_num)
        from_server = tff.federated_broadcast(from_server)
        control_output = tff.federated_map(
            control_computation, (federated_dataset_single, 
                                    client_state, from_server))
        c = tff.federated_broadcast(tff.federated_mean(
            control_output.w_delta, weight=control_output.client_weight
        ))
        @tff.tf_computation(client_output_type.w_delta,
                        client_output_type.w_delta)
        def compute_control_input(c, c_i):
            correction = tf.nest.map_structure(lambda a, b: a - b, c, c_i)
            # if we are using SCAFFOLD then use the correction, otherwise 
            # we just let the correction be zero.
            return tf.cond(tf.constant(control,dtype=tf.bool), 
                lambda: correction, 
                lambda: tf.nest.map_structure(tf.zeros_like, correction))
        # the collection of gradient corrections
        corrections = tff.federated_map(compute_control_input, 
                            (c, control_output.cs))
        client_outputs = tff.federated_map(
            client_computation, (federated_dataset, from_server, client_state, 
            corrections)
        )
        w_delta = tff.federated_mean(client_outputs.w_delta,
                        weight=client_outputs.client_weight)
        server_state = tff.federated_map(server_computation, 
                        (server_state, w_delta))
        return (server_state, client_outputs.client_state)

    return tff.templates.IterativeProcess(initialize_fn(), run_one_round)






        


    