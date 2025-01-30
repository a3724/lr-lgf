from typing import NamedTuple
import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp

from .base_utils import GlobalRegistry

import jax
import jax.numpy as jnp
import optax
import numpy as np


class Regularizers:
    @staticmethod
    def l2_regularizer(params):
        """
        Compute L2 regularization for model parameters.

        Args:
            params: Model parameters

        Returns:
            float: L2 regularization value
        """
        params_flat, _ = jax.flatten_util.ravel_pytree(params)
        diff = params_flat @ params_flat.T
        return 0.5 * diff

    @staticmethod
    def laplace_regularizer(params, mPi: jax.Array, Pi_t: list, mode):
        """
        Compute Laplace regularization for model parameters.

        Args:
            params: Model parameters
            mPi: Mean parameter vector
            Pi_t: List of transformation matrices

        Returns:
            tuple: (regularization value, (first term, second term, third term))
        """
        params_flat = jax.flatten_util.ravel_pytree(params)[0][None, :]
        first = params_flat * Pi_t[0] @ params_flat.T
        second = ((params_flat @ Pi_t[1]) @ Pi_t[2]) @ Pi_t[1].T @ params_flat.T
        third = 2 * params_flat @ mPi
        fourth = jnp.sum(mode * mPi.T)
        #result = 0.5 * (first + second - third)
        result = 0.5 * (first + second - third + fourth)
        return jnp.sum(result), (first, second, third)

    def compute_diag_fim(params: hk.Params, data):
        loss_ = GlobalRegistry.get_loss_()
        grad_log_likelihood = jax.grad(loss_, argnums=0)
        grads = grad_log_likelihood(params, data)
        grads_flat, _ = jax.flatten_util.ravel_pytree(grads)
        return grads_flat ** 2

    def FIM_samples(params: hk.Params, data, fim_samples=1):
        config = GlobalRegistry.get_config()
        compute_diag_fim_batched = GlobalRegistry.get_compute_diag_fim_batched()
        params_flat, _ = jax.flatten_util.ravel_pytree(params)
        FIM = np.zeros_like(params_flat)
        # Compute FIM
        for _ in range(fim_samples):
            batch = next(data)
            FIM[:] += compute_diag_fim_batched(params, batch).mean(axis=0)
        # Normalize
        FIM /= fim_samples * config.BATCH_SIZE_HES
        jax.debug.print('FIM_last, {fim}', fim=FIM)
        return FIM

    ### EWC - REGULARISER

    def fisher_reg(params: hk.Params, prior_mean: jax.Array, prior_prec: jax.Array):
        params_flat, _ = jax.flatten_util.ravel_pytree(params)
        diff = 0
        for ti in range(0, prior_mean.shape[0]):
            diff_flat = params_flat - prior_mean[ti]
            temp = prior_prec[ti, :] * diff_flat
            diff += diff_flat.T @ temp
        return 0.5 * diff


class AdditionalFunctions:
    @staticmethod
    def calculate_loss_alone(logits, batch, train):
        """
        Calculate the base loss without regularization.
        """
        imgs, labels, metadata = batch
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss

    @staticmethod
    def compute_GGN(model, state, batch, train=False):
        """
        Compute the Generalized Gauss-Newton matrix.

        Args:
            model: The neural network model
            state: Current training state
            batch: Input batch (images, labels, metadata)
            train: Boolean indicating training mode

        Returns:
            tuple: (Jacobian, Hessian)
        """

        def J_wrapper_function(flattened_weights, func_to_unflatten, image, train):
            unflattened_weights = func_to_unflatten(flattened_weights)
            outs = model.apply({'params': unflattened_weights, 'batch_stats': state.batch_stats},
                               image,
                               train=train,
                               mutable=['batch_stats'] if train else False)
            logits, new_model_state = outs if train else (outs, None)
            return logits

        images, labels, metadata = batch
        flat_params, tree_str = jax.flatten_util.ravel_pytree(state.params)
        J = jax.jacobian(J_wrapper_function)(flat_params, tree_str, images, train)

        outs = model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                           images,
                           train=train,
                           mutable=['batch_stats'] if train else False)
        logits, new_model_state = outs if train else (outs, None)
        H = jax.hessian(AdditionalFunctions.calculate_loss_alone)(logits, (images, labels, metadata), train)
        H = H[np.arange(0, H.shape[0]), :, np.arange(0, H.shape[0]), :]
        H = np.array(H)

        return J, H

    @staticmethod
    def set_Q(theta_star):
        config = GlobalRegistry.get_config()
        Q = np.zeros((1, theta_star.size))

        # Constants for Camelyon architecture
        bn = 32 + 32 + 16 + 16 + 4 + 4
        c0_kernel = bn + 864
        c0_bias = c0_kernel + 32
        c1_kernel = c0_bias + 4608
        c1_bias = c1_kernel + 16
        c2_kernel = c1_bias + 576
        c2_bias = c2_kernel + 4
        d0_kernel = c2_bias + 128
        d0_bias = d0_kernel + 8
        d1_kernel = d0_bias + 16

        # Set Q values for each layer
        Q[0, :bn] = 0
        Q[0, bn:c0_kernel] = config.Q_c0_kernel * (theta_star[bn:c0_kernel] ** 2).mean()
        Q[0, c0_kernel:c0_bias] = config.Q_c0_bias * (theta_star[c0_kernel:c0_bias] ** 2).mean()
        Q[0, c0_bias:c1_kernel] = config.Q_c1_kernel * (theta_star[c0_bias:c1_kernel] ** 2).mean()
        Q[0, c1_kernel:c1_bias] = config.Q_c1_bias * (theta_star[c1_kernel:c1_bias] ** 2).mean()
        Q[0, c1_bias:c2_kernel] = config.Q_c1_kernel * (theta_star[c1_bias:c2_kernel] ** 2).mean()
        Q[0, c2_kernel:c2_bias] = config.Q_c2_bias * (theta_star[c2_kernel:c2_bias] ** 2).mean()
        Q[0, c2_bias:d0_kernel] = config.Q_d0_kernel * (theta_star[c2_bias:d0_kernel] ** 2).mean()
        Q[0, d0_kernel:d0_bias] = config.Q_d0_bias * (theta_star[d0_kernel:d0_bias] ** 2).mean()
        Q[0, d0_bias:d1_kernel] = config.Q_d1_kernel * (theta_star[d0_bias:d1_kernel] ** 2).mean()
        Q[0, d1_kernel:] = config.Q_d1_bias * (theta_star[d1_kernel:] ** 2).mean()
        if config.Q_ALL != 'None':
            Q[0, :] = config.Q_ALL

        return Q

'''
def all_trans(params, x, transformations):
    '''
    # Applies all the transformations of the model to the input

    #Args:
    #    params (dict): model parameters
    #    x (jax.Array): input
    #    transformations (list of function): list of transformations

    ##Returns:
    #    x (jax.Array): transformed output
    #    list_int (list): list of intermediate outputs of each transformation
'''
    list_int = []
    for ti in transformations:
        x = ti(params,None, x)
        list_int.append(x)
    return x, list_int

def last_trans(sx_1, batch):
    '''
    #Computes the last transformation of the model
'''
    config = GlobalRegistry.get_config()
    labels = jax.nn.one_hot(batch.label, config.NUM_CLASSES)
    return labels * jax.nn.log_softmax(sx_1)

def update_lists(params, batch):
    '''
    #Creates the list of transformations and gradients for each transformation.

    #Args:
    #    params (dict): model parameters
    #    batch (Batch): batch of data

    #Returns:
    #    list_s (list): list of intermediate outputs of each transformation of the layer a(input)
    #    list_a (list): list of intermediate outputs of each transformation of the layer s(intput)
    #    list_g (list): list of gradients of each transformation of the layer s(input)
'''
    net_batched = GlobalRegistry.get_net_batched()
    loss_batched = GlobalRegistry.get_loss_batched()

    list_s, list_a = all_trans(params, batch.image, net_batched)[1][1::2], all_trans(params, batch.image, net_batched)[1][0::2]
    list_a.append(last_trans(list_s[-1], batch))

    list_a = [np.concatenate((i, np.ones((i.shape[0], 1))), axis=1) for i in list_a]
    list_g = [jax.grad(loss_batched, argnums=1)(params, list_s[k] , batch.label, L_start=2+2*k) for k in range(len(list_s))]
    return list_s, list_a, list_g

def compute_KFAC_matrix(L, list_a, list_g):
    '''
    #Computes the KFAC block-diagonal i.e. each consists of two matrices - one is the expected value of the outer product of the activations and the other the expected value of the outer product of the gradients.
'''
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
'''
# FULL REGULARIZER STEP
def kalman_step(task_id, J,H, cnn_trainer, batch_statss, list_of_Pts,flt, lam_task, Qs, thetas):
    ''' Compute GGN. '''
    config = GlobalRegistry.get_config()
    theta_star, unflatten_func = jax.flatten_util.ravel_pytree(cnn_trainer.state.params)

    # Predict
    if (config.NAME_RES == 'inferred_params' and task_id %2==0) or config.NAME_RES != 'inferred_params':
        print('Add_low', task_id)
        flt.add_low(J, H, lam_task, task_id)
    Q = AdditionalFunctions.set_Q(theta_star)

    if task_id == 0:
        Qs = np.zeros((config.T, theta_star.size))
        thetas = np.zeros((config.T, theta_star.size))

    Qs[task_id, :] = Q
    thetas[task_id, :] = theta_star
    batch_statss.append(cnn_trainer.state.batch_stats)

    list_of_Pts.append(flt.Pi_t)

    flt.compute_inv_sum_diag_dlr(Q)
    flt.update_mPi(theta_star)

    return flt, theta_star, unflatten_func, batch_statss, Qs, thetas

'''
def ewc_step(task_id, train_datasets_hes,state,prior_prec, prior_mean, flat_params_init, fim_samples=1):
    '''
    # Compute the Fisher Information Matrix (FIM) and update the prior precision and mean.
'''
    new_FIM = FIM_samples(state.params, train_datasets_hes[task_id], fim_samples=fim_samples)
    prior_prec[task_id+1, :] = new_FIM
    #prior_prec[0, :] =  np.zeros((1, flat_params_init.shape[0])) # Removes the initial L2 from the general regularizer
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
'''