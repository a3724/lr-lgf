from typing import NamedTuple
import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp

from .base_utils import GlobalRegistry, Batch



### GENERAL

def l2_regulariser(params: hk.Params, params_before: hk.Params):
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    params_bef_flat, _ = jax.flatten_util.ravel_pytree(params_before)
    diff_flat = params_flat - params_bef_flat
    diff = diff_flat @ diff_flat
    return 0.5*diff


### OURS 

# For GGN
def loss_no_reg_(logits, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian.
    """
    config = GlobalRegistry.get_config()
    labels = jax.nn.one_hot(batch.label, config.NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return -log_likelihood

def loss_no_reg(logits, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularized by Laplace regularizer with the full Hessian."""
    res = jnp.mean(loss_no_reg_(logits, batch))
    return res

def compute_GGN(params, batch):
    """
    A function that computes the GGN - (df/§dw)^T @ (d^2L/df^2) @ (df/dw):
    * Compute the Jacobian - flatten the parameters anf run with the wrapper function to obtain df/dw.
    * Compute the Hessian - we obtain d^2L/df^2.

    """
    network = GlobalRegistry.get_network()
    flat_params, tree_str = jax.flatten_util.ravel_pytree(params)
    J = jax.jacobian(J_wrapper_function)(flat_params, tree_str, batch.image)

    logits = network.apply(params, batch.image)
    H = jax.hessian(loss_no_reg)(logits, batch)

    return J, H


def J_wrapper_function(flattened_weights, func_to_unflatten, image):
    network = GlobalRegistry.get_network()
    unflattened_weights = func_to_unflatten(flattened_weights)
    return network.apply(unflattened_weights, image)

def set_Q(theta_star):
    config = GlobalRegistry.get_config()
    LOWER_LAYERS = config.NUM_MID+config.INPUT_IMAGE_SIZE*config.INPUT_IMAGE_SIZE*config.NUM_MID
    Q = np.zeros((1, theta_star.size))

    Q[:, :config.NUM_MID] = config.ALPHA_LOW_B*(theta_star[:config.NUM_MID]**2).mean() #low bias
    Q[:, config.NUM_MID:LOWER_LAYERS] = config.ALPHA_LOW_W*(theta_star[config.NUM_MID:LOWER_LAYERS]**2).mean() #low weight
    if config.NUM_LAYERS == 2:
        Q[:, LOWER_LAYERS:(LOWER_LAYERS + config.NUM_MID)] = config.ALPHA_MID_W * theta_star[LOWER_LAYERS:(LOWER_LAYERS + config.NUM_MID)]  # mid bias
        Q[:, (LOWER_LAYERS + config.NUM_MID):(LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID)] = config.ALPHA_MID_B * theta_star[(LOWER_LAYERS + config.NUM_MID):(LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID)]  # mid weight
        Q[:, (LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID):(LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID + config.NUM_CLASSES)] = config.ALPHA_HIGH_W*theta_star[(LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID):(LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID + config.NUM_CLASSES)] #high bias
        Q[:, (LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID + config.NUM_CLASSES):] = config.ALPHA_HIGH_B*theta_star[(LOWER_LAYERS + config.NUM_MID + config.NUM_MID*config.NUM_MID + config.NUM_CLASSES):] #high weight
    elif config.NUM_LAYERS == 1:
        Q[:, LOWER_LAYERS:(LOWER_LAYERS + config.NUM_CLASSES)] = config.ALPHA_HIGH_W * theta_star[LOWER_LAYERS:(LOWER_LAYERS + config.NUM_CLASSES)]  # high bias
        Q[:, (LOWER_LAYERS + config.NUM_CLASSES):] = config.ALPHA_HIGH_B * theta_star[(LOWER_LAYERS + config.NUM_CLASSES):]  # high weight
    return Q

### OURS - REGULARISER

def laplace_regulariser(params, mPi: jax.Array, Pi_t:  list, mode):
    params_flat = jax.flatten_util.ravel_pytree(params)[0][None, :]
    first = params_flat * Pi_t[0] @ params_flat.T
    second = ((params_flat @ Pi_t[1]) @ Pi_t[2]) @ Pi_t[1].T @ params_flat.T
    third = 2 * params_flat @ mPi
    fourth = jnp.sum(mode * mPi.T)
    result = 0.5 * (first + second - third)
    #result = 0.5 * (first + second - third + fourth)
    return jnp.sum(result)

### EWC

def compute_diag_fim(params: hk.Params, data):
    loss_ = GlobalRegistry.get_loss_()
    grad_log_likelihood = jax.grad(loss_, argnums=0)
    grads = grad_log_likelihood(params, data)
    grads_flat, _ = jax.flatten_util.ravel_pytree(grads)
    return grads_flat**2

def FIM_samples(params: hk.Params, data,fim_samples=1):
    config = GlobalRegistry.get_config()
    compute_diag_fim_batched = GlobalRegistry.get_compute_diag_fim_batched()
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    FIM = np.zeros_like(params_flat)
    # Compute FIM
    for _ in range(fim_samples):
        batch = next(data)
        FIM[:] += compute_diag_fim_batched(params, batch).mean(axis=0)
    # Normalize
    FIM /= fim_samples
    jax.debug.print('FIM_last, {fim}', fim=FIM)
    return FIM

### EWC - REGULARISER

def fisher_reg(params: hk.Params, prior_mean: jax.Array, prior_prec: jax.Array):
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    diff = 0
    for ti in range(0,prior_mean.shape[0]):
        diff_flat = params_flat - prior_mean[ti]
        temp = prior_prec[ti,:] * diff_flat
        diff += diff_flat.T @ temp
    return 0.5*diff

### OSLA
def all_trans(params, x, transformations):
    '''
    Applies all the transformations of the model to the input

    Args:
        params (dict): model parameters
        x (jax.Array): input
        transformations (list of function): list of transformations

    Returns:
        x (jax.Array): transformed output
        list_int (list): list of intermediate outputs of each transformation
    '''
    list_int = []
    for ti in transformations:
        x = ti(params,None, x)
        list_int.append(x)
    return x, list_int

def last_trans(sx_1, batch):
    '''Computes the last transformation of the model'''
    config = GlobalRegistry.get_config()
    labels = jax.nn.one_hot(batch.label, config.NUM_CLASSES)
    return labels * jax.nn.log_softmax(sx_1)

def update_lists(params, batch):
    '''
    Creates the list of transformations and gradients for each transformation.

    Args:
        params (dict): model parameters
        batch (Batch): batch of data

    Returns:
        list_s (list): list of intermediate outputs of each transformation of the layer a(input)
        list_a (list): list of intermediate outputs of each transformation of the layer s(intput)
        list_g (list): list of gradients of each transformation of the layer s(input)
    '''
    net_batched = GlobalRegistry.get_net_batched()
    loss_batched = GlobalRegistry.get_loss_batched()

    list_s, list_a = all_trans(params, batch.image, net_batched)[1][1::2], all_trans(params, batch.image, net_batched)[1][0::2]
    list_a.append(last_trans(list_s[-1], batch))

    list_a = [np.concatenate((i, np.ones((i.shape[0], 1))), axis=1) for i in list_a]
    list_g = [jax.grad(loss_batched, argnums=1)(params, list_s[k] , batch.label, L_start=2+2*k) for k in range(len(list_s))]
    return list_s, list_a, list_g

def compute_KFAC_matrix(L, list_a, list_g):
    '''Computes the KFAC block-diagonal i.e. each consists of two matrices - one is the expected value of the outer product of the activations and the other the expected value of the outer product of the gradients.'''
    config = GlobalRegistry.get_config()
    def expected_val(listx, index1, index2):
        return listx[index1].T @ listx[index2] / config.BATCH_SIZE
    kfac = [[expected_val(list_a, i, i), expected_val(list_g, i, i)] for i in range(len(list_a)-1)]
    return kfac

### OSLA - REGULARISER

def kfac_reg(params: hk.Params, prior_mean: hk.Params, prior_prec: hk.Params):
    result = 0
    for i, ki in enumerate(params.keys()):
        diff_w = (params[ki]['w'] - prior_mean[ki]['w'])
        diff_b = (params[ki]['b'] - prior_mean[ki]['b'])
        diff = jnp.concatenate((diff_w, diff_b[None, :]), axis=0)
        res = prior_prec[i][1] @ diff.T @ prior_prec[i][0]
        result += np.sum(res.reshape((1,-1)) @ diff.T.reshape((1,-1)).T)
    return result

# FULL REGULARIZER STEP

def kalman_step(task_id, train_datasets_hes,state, flt, lam_task):
    ''' Compute GGN. '''
    compute_GGN_batched = GlobalRegistry.get_compute_GGN_batched()
    train_data = next(train_datasets_hes[task_id])
    J, H = compute_GGN_batched(state.params, train_data)
    theta_star, _ = jax.flatten_util.ravel_pytree(state.params)

    ''' Kalman estimation. '''
    # Update
    flt.add_low(J, H, lam_task, task_id)
    temp_Pi0, temp_Pi1, temp_Pi2 = flt.Pi_t[0], flt.Pi_t[1], flt.Pi_t[2]
    # Predict
    Q = set_Q(theta_star)
    if np.any(Q) != 0:
        flt.compute_inv_sum_diag_dlr(Q)
    flt.update_mPi(theta_star)
    return flt, theta_star


def ewc_step(task_id, train_datasets_hes,state,prior_prec, prior_mean, flat_params_init, fim_samples=1):
    '''
    Compute the Fisher Information Matrix (FIM) and update the prior precision and mean.
    '''
    new_FIM = FIM_samples(state.params, train_datasets_hes[task_id], fim_samples=fim_samples)
    prior_prec[task_id+1, :] = new_FIM
    prior_prec[0, :] =  np.zeros((1, flat_params_init.shape[0])) # Removes the initial L2 from the general regularizer
    prior_mean[task_id+1,:], _ = jax.flatten_util.ravel_pytree(state.params)
    return prior_prec, prior_mean

def osla_step(state, train_data_hes, prior_prec, lam_task):
    _, list_a, list_g = update_lists(state.params, train_data_hes)
    new_KFAC = compute_KFAC_matrix(len(list_a), list_a, list_g)
    prior_prec2 = []
    for i in range(len(new_KFAC)):
        prior_prec2.append([prior_prec[i][0] + lam_task*new_KFAC[i][0], prior_prec[i][1] + new_KFAC[i][1]])
    prior_prec = prior_prec2
    prior_mean = state.params
    return prior_prec, prior_mean