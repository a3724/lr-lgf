from typing import NamedTuple, Any
import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp
import tqdm
import os
import wandb
from flax import linen as nn
from flax.training import train_state, checkpoints
from collections import defaultdict
from .base_utils import  GlobalRegistry, Batch
from .regularizer_utils import laplace_regulariser, fisher_reg, kalman_step, ewc_step, l2_regulariser, kfac_reg, all_trans, osla_step


class TrainingState(NamedTuple):
    """
    Represents the training state containing model parameters and optimizer state.
    """
    params: hk.Params
    opt_state: optax.OptState

def optimizer_fn(weight_decay: bool, weight_decay_val: float, lr: float):
    """
    Constructs an optimizer with optional weight decay.

    Args:
        weight_decay (bool): Whether to apply weight decay.
        weight_decay_val (float): Weight decay value if applicable.
        lr (float): Learning rate.

    Returns:
        optax.GradientTransformation: The optimizer configuration.
    """
    if weight_decay:
        tx = optax.chain(
            optax.add_decayed_weights(weight_decay_val),
            optax.adam(learning_rate=lr)
        )
    else:
        tx = optax.adam(learning_rate=lr)
    return tx

### MODEL


class MLPModule(hk.Module):
    """
    A flexible MLP (Multi-Layer Perceptron) module for classification with dynamic hidden layers.

    Args:
        num_mid (int): Number of units in each hidden layer.
        num_classes (int): Number of output classes.
        num_layers (int): Number of hidden layers.
    """

    def __init__(self, num_mid: int, num_classes: int, num_layers: int, name=None):
        super().__init__(name=name)
        self.num_mid = num_mid
        self.num_classes = num_classes
        self.num_layers = num_layers

    def __call__(self, images):
        """
        Forward pass through the MLP.

        Args:
            images (jnp.ndarray): Input image data.

        Returns:
            jnp.ndarray: Output logits for classification.
        """
        x = images.astype(jnp.float32) / 255.
        x = jnp.reshape(x, (-1,))

        layers = []
        for _ in range(self.num_layers):
            layers.append(hk.Linear(self.num_mid))
            layers.append(jax.nn.relu)

        # Add the output layer
        layers.append(hk.Linear(self.num_classes))

        mlp = hk.Sequential(layers)
        return mlp(x)


def net_fn(images):
    config = GlobalRegistry.get_config()
    mlp_module = MLPModule(num_mid=config.NUM_MID, num_classes=config.NUM_CLASSES, num_layers=config.NUM_LAYERS)
    return mlp_module(images)

class NormalizeAndReshape(hk.Module):
    ''' Module for normalizing and reshaping the input in the OSLA model.'''
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.
        x = jnp.reshape(x, (-1,))
        return x


def f():
    config = GlobalRegistry.get_config()

    pre = NormalizeAndReshape()
    s0 = hk.Linear(config.NUM_MID)
    a0 = jax.nn.relu
    s1 = hk.Linear(config.NUM_MID)
    a1 = jax.nn.relu
    s2 = hk.Linear(config.NUM_CLASSES)

    def init(x):
        x = pre(x)
        x = s0(x)
        x = a0(x)
        x = s1(x)
        x = a1(x)
        x = s2(x)
        return x

    return init, (pre, s0, a0, s1, a1, s2)

def create_dynamic_mlp_with_module():
    '''MLP for OSLA code, returns also each layer.'''
    config = GlobalRegistry.get_config()

    class AccessibleMLP(MLPModule):
        def get_components(self):
            pre = NormalizeAndReshape()
            layers = []
            activations = []

            for _ in range(self.num_layers):
                layers.append(hk.Linear(self.num_mid))
                activations.append(jax.nn.relu)
            layers.append(hk.Linear(self.num_classes))

            # Create components tuple
            all_components = [pre]
            for i in range(len(layers) - 1):
                all_components.extend([layers[i], activations[i]])
            all_components.append(layers[-1])
            return tuple(all_components)

        def __call__(self, images):
            components = self.get_components()
            pre = components[0]
            x = pre(images)

            for i in range(1, len(components) - 1, 2):
                x = components[i](x)  # Linear layer
                x = components[i + 1](x)  # Activation
            x = components[-1](x)
            return x

    mlp = AccessibleMLP(
        num_mid=config.NUM_MID,
        num_classes=config.NUM_CLASSES,
        num_layers=config.NUM_LAYERS
    )

    return mlp.__call__, mlp.get_components()

### LOSS

def loss_(params: hk.Params, batch: Batch) -> jax.Array:
    network = GlobalRegistry.get_network()
    config = GlobalRegistry.get_config()
    logits = network.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, config.NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
    return -log_likelihood

def loss_net_xL_(params, x, label, L_start):
    '''
    Computes the transformed input.

    Args:
        params (hk.Params): the parameters of the model
        x (): the input that will be transformed
        label (): the labels of the inputs
        L_start (int): defines which layer the transformation starts form

    Returns:
        (float): negative log-likelihood
    '''
    net_batched = GlobalRegistry.get_net_batched()
    config = GlobalRegistry.get_config()
    s = x
    for i in range(L_start, len(net_batched)-1, 2):
        a = net_batched[i](params, None, s)
        s = net_batched[i + 1](params, None, a)
    labels = jax.nn.one_hot(label, config.NUM_CLASSES)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(s))
    return -log_likelihood


def loss(params: hk.Params, batch: Batch, mPi: jax.Array, Pi_t: list, lam, mode=0) -> jax.Array:
    config = GlobalRegistry.get_config()
    loss_batched = GlobalRegistry.get_loss_batched()

    if config.METHOD == 'ours':
        res_temp = jnp.mean(loss_batched(params, batch))
        reg = laplace_regulariser(params, mPi, Pi_t, mode)
        res = res_temp + lam*reg
    elif config.METHOD == 'ewc':
        res_temp = jnp.mean(loss_batched(params, batch))
        reg = fisher_reg(params, mPi, Pi_t)
        res = res_temp + lam*reg
    elif config.METHOD == 'baseline':
        res_temp = jnp.mean(loss_batched(params, batch))
        reg = l2_regulariser(params, mPi)
        res = res_temp + lam*reg
    elif config.METHOD == 'osla':
        res_temp = loss_net_xL_(params, batch.image, batch.label, L_start = 0)
        kfac_vc = kfac_reg(params, mPi, Pi_t)
        res = res_temp + lam*kfac_vc
        kfac_vc = kfac_vc.astype(jnp.float32)
        return res, (kfac_vc, res_temp)
    else:
        res_temp = jnp.mean(loss_batched(params, batch))
        reg = 0
        res = res_temp
    return res, (reg, res_temp)

@jax.jit
def update(state: TrainingState, batch: Batch, mPi, Pi_t, lam, mode=0) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    optimizer = GlobalRegistry.get_optimizer()
    vals, grads  = jax.value_and_grad(loss, argnums=0, has_aux=True)(state.params, batch, mPi, Pi_t, lam, mode)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainingState(params, opt_state), vals

@jax.jit
def evaluate(params: hk.Params, batch: Batch) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    config = GlobalRegistry.get_config()
    net_batched = GlobalRegistry.get_net_batched()
    if config.METHOD == 'osla':
        logits = all_trans(params, batch.image, net_batched)[0]
    else:
        logits = net_batched(params, batch.image)

    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch.label)


### TRAINING
def training_loop(state, train_datasets, train_datasets_hes, valid_datasets,  acc_matrix, avg_test_accs, flat_params_init,
                  flt=None, prior_prec=None, prior_mean=None, save_params_Pi_t=0):
    '''
    Training loop for the given configuration.
    '''
    config = GlobalRegistry.get_config()
    which = 'test' if config.TEST == 1 else 'valid'
    track_train_loss = 1
    track_valid_loss = 1
    mode = 0
    if save_params_Pi_t:
        jnp.save(os.path.join(config.DIR_RES, 'model_%03i.npy' % (0)),
                 hk.data_structures.to_mutable_dict(state.params))
        np.savez(os.path.join(config.DIR_RES, 'Pi_t_%03i.npz' % (0)), *flt.Pi_t)
        print('Saved ' + os.path.join(config.DIR_RES, 'Pi_t_%03i.npz' % (0)))
    # or (config.METHOD == 'ewc' and task_id == 0 and config.LAMBDA != 0)\

    for task_id in range(config.T):

        if (config.METHOD == 'ours' and task_id >= 0 and config.LAMBDA == 0)\
            or (config.METHOD == 'baseline' and task_id > 0)\
            or (config.METHOD == 'ewc' and task_id > 0) or (config.METHOD == 'osla' and task_id > 0 and config.LAMBDA == 0):
            print('The condition was met for lambdas')
            LAM_TASK = config.LAMBDA
        elif (config.METHOD == 'baseline' and task_id == 0):
            LAM_TASK = config.LAMBDA_INIT
        else:
            LAM_TASK = 1
        print('Training ', LAM_TASK, config.METHOD, task_id, config.LAMBDA, config.LAMBDA_INIT)

        # Train
        for step in tqdm.trange(config.EPOCH):
            train_data = next(train_datasets[task_id])
            # Do SGD on a batch of training examples.
            if config.METHOD == 'ours':
                state, vals = update(state, train_data, flt.mPi, flt.Pi_t, LAM_TASK, mode)
            elif config.METHOD == 'ewc':
                state, vals = update(state, train_data, prior_mean, prior_prec, LAM_TASK)
            elif config.METHOD == 'baseline':
                state, vals = update(state, train_data, prior_mean, None, LAM_TASK)
            elif config.METHOD == 'osla':
                state, vals = update(state, train_data, prior_mean, prior_prec, LAM_TASK)
            else:
                raise 'Error'
            if track_train_loss == 1:
                wandb.log({"train/loss/" + str(task_id): vals[1][1]})
                wandb.log({"train/reg/" + str(task_id): vals[1][0]})
            if track_valid_loss == 1:
                valid_data = next(valid_datasets[task_id])
                if config.METHOD == 'ours':
                    vals = loss(state.params, valid_data, flt.mPi, flt.Pi_t, LAM_TASK, mode)
                elif config.METHOD == 'ewc':
                    vals = loss(state.params, valid_data, prior_mean, prior_prec, LAM_TASK)
                elif config.METHOD == 'osla':
                    vals = loss(state.params, valid_data, prior_mean, prior_prec, LAM_TASK)
                elif config.METHOD == 'baseline':
                    vals = loss(state.params, valid_data, prior_mean, None, LAM_TASK)
                else:
                    raise 'Error'
                wandb.log({which+"/loss/" + str(task_id): vals[0]})


        # Evaluate
        for ti in range(task_id + 1):
            for step_ev in range(config.HOW_MANY_EVALS):
                valid_data = next(valid_datasets[ti])
                acc_matrix[task_id, ti] += evaluate(state.params, valid_data)
            acc_matrix[task_id, ti] /= config.HOW_MANY_EVALS
            print('Current task:', task_id, 'Evaluated on:', ti)
            print(f'Test Accuracy (%): {acc_matrix[task_id, ti]:.2f}).')

        avg_test_acc_t = jnp.mean(acc_matrix[task_id, :(task_id + 1)])
        avg_test_accs.append(avg_test_acc_t)
        for ti in range(task_id + 1):
            wandb.log({which+"/acc_per_task/" + str(ti): acc_matrix[task_id, ti]})
        wandb.log({which+"/acc_avr/": avg_test_acc_t})
        print(f'Average Accuracy (%): {avg_test_acc_t:.2f}).')

        # Update the Hessian

        LAM_TASK = config.LAMBDA
        print('Update ', LAM_TASK)
        if config.METHOD == 'ours':
            flt, mode = kalman_step(task_id, train_datasets_hes,state, flt, LAM_TASK)
        elif config.METHOD == 'ewc':
            prior_prec, prior_mean = ewc_step(task_id, train_datasets_hes,state,prior_prec, prior_mean, flat_params_init, fim_samples=1)
        elif config.METHOD == 'osla':
            prior_prec, prior_mean = osla_step(state, next(train_datasets_hes[ti]), prior_prec, LAM_TASK)
        # Save params and Pi_t
        if save_params_Pi_t:
            jnp.save(os.path.join(config.DIR_RES, 'model_%03i.npy' % (task_id + 1)),
                 hk.data_structures.to_mutable_dict(state.params))
            np.savez(os.path.join(config.DIR_RES, 'Pi_t_%03i.npz' % (task_id + 1)), *flt.Pi_t)
            print('Saved '+os.path.join(config.DIR_RES, 'Pi_t_%03i.npz' % (task_id + 1)))

    return state, acc_matrix, avg_test_accs





