import tensorflow as tf
import tensorflow_federated as tff
import functools
import attr
import collections
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
@tf.function
def zero_all_if_any_non_finite(structure):
  """Zeroes out all entries in input if any are not finite.
  Args:
    structure: A structure supported by tf.nest.
  Returns:
     A tuple (input, 0) if all entries are finite or the structure is empty, or
     a tuple (zeros, 1) if any non-finite entries were found.
  """
  flat = tf.nest.flatten(structure)
  if not flat:
    return (structure, tf.constant(0))
  flat_bools = [tf.reduce_all(tf.math.is_finite(t)) for t in flat]
  all_finite = functools.reduce(tf.logical_and, flat_bools)
  if all_finite:
    return (structure, tf.constant(0))
  else:
    return (tf.nest.map_structure(tf.zeros_like, structure), tf.constant(1))

@tf.function
def project_to_simplex(v):
    n_features = v.shape[0]
    u = tf.sort(v, direction='DESCENDING')
    cssv = tf.math.cumsum(u) - 1.
    ind = tf.cast(tf.range(n_features) + 1, dtype=tf.float32)
    cond = u - cssv / ind > 0 
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / tf.cast(rho, dtype=tf.float32)
    w = tf.maximum(v - theta, tf.zeros_like(v))
    w = w/tf.reduce_sum(w)
    return w

def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)

@tf.function
def server_update(model,
                    meta_update_interval,
                    server_optimizer,
                    server_state,
                    weights_delta):
    model_weights = _get_weights(model)
    tff.utils.assign(model_weights, server_state.w)

    weights_delta, has_non_finite_weight = (
        zero_all_if_any_non_finite(weights_delta))
    grads_and_vars = [
      (-1.0 * x, v) for x, v in zip(weights_delta, model_weights.trainable)
    ]
    server_optimizer.apply_gradients(grads_and_vars)

    numrounds = server_state.round_num
    # if it is time to update the meta-iterate
    round_mod = tf.math.floormod(numrounds,meta_update_interval,name=None)
    meta_w = tf.cond(tf.math.equal(round_mod, 0),
        lambda: model_weights,
        lambda: server_state.meta_w)
    return tff.utils.update_state(
        server_state,
        w = model_weights,
        meta_w = meta_w,
        round_num=server_state.round_num + 1.0
    )

#@tf.function
def client_update(model, p, p_even, client_optimizer, tau, p_size, shift_reg,
                    reg, meta_update_interval, client_lr, dataset, client_state,
                    from_server, w_correction):
    model_weights = _get_weights(model)
    tff.utils.assign(model_weights, from_server.w)
    tff.utils.assign(p, client_state.p)
    meta_w = from_server.meta_w
    meta_p = client_state.meta_p
    num_examples = tf.constant(0, dtype=tf.int32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.compat.v1.losses.Reduction.NONE)
    for batch in dataset:
        with tf.GradientTape(persistent=True) as tape:
            output = model.forward_pass(batch)
            batch_losses = loss_fn(y_true = batch.get('y'),
                y_pred = output.predictions)
            weighted_losses = p * batch_losses    
            loss = tf.reduce_sum(weighted_losses)
            for i in range(len(model_weights.trainable)):
                # catalyst for w
                loss += tau*tf.nn.l2_loss(
                    model_weights.trainable[i] - meta_w.trainable[i])
                loss += reg*tf.nn.l2_loss(model_weights.trainable[i])
            # catalyst for p
            loss -= tau*tf.nn.l2_loss(meta_p - p)
            # lagrangian for deviation away from 1/n
            loss -= shift_reg*tf.nn.l2_loss(p_even - p)
            
        # optimize the w's
        grads = tape.gradient(loss, model_weights.trainable)
        corrected_grads = tf.nest.map_structure(lambda a,b: a + b/client_lr,
                                grads, w_correction) ## the SCAFFOLD correction
        grads_and_vars = zip(corrected_grads, model_weights.trainable)
        client_optimizer.apply_gradients(grads_and_vars)
        
        # optimize the p's
        grads = tape.gradient(loss, p)
        grads_and_vars = [(-grads, p)]
        client_optimizer.apply_gradients(grads_and_vars)
        # project to the simplex
        tff.utils.assign(p, project_to_simplex(p))
        # increment the examples accessed 
        num_examples += tf.shape(output.predictions)[0]
    aggregated_outputs = model.report_local_outputs()
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          from_server.w.trainable)
    weights_delta, has_non_finite_weight = (
        zero_all_if_any_non_finite(weights_delta))
    
    if has_non_finite_weight > 0:
        client_weight = tf.constant(0, dtype=tf.float32)
    else:
        client_weight = tf.cast(num_examples, dtype=tf.float32)
    # if it is time to update the meta-iterate
    round_num = from_server.round_num
    round_mod = tf.math.floormod(round_num,meta_update_interval,name=None)
    
    meta_p = tf.cond(tf.math.equal(round_mod, 0),
        lambda: p,
        lambda: meta_p)
    
    client_state = tff.utils.update_state(
        client_state,
        meta_p=meta_p,
        p=p
    )
    return ClientOutput(
        weights_delta, client_weight, client_state, aggregated_outputs,
        collections.OrderedDict([('num_examples', num_examples)])
    )
    
    


    

    
