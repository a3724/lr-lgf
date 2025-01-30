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
from .base_utils import GlobalRegistry, TrainState
from .regularizer_utils import kalman_step
from .DiagLowRank import Diag_LowRank

import tensorflow_datasets as tfds


import jax
import jax.numpy as jnp
import optax
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import wandb
import os
from flax.training import checkpoints
import optax
from typing import Any

from flax import linen as nn
from .regularizer_utils import Regularizers, AdditionalFunctions

class CNN(nn.Module):
    num_classes : int
    @nn.compact
    def __call__(self, x, train=True):

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.max_pool(x, window_shape=(4, 4), strides=(4, 4))

        x = nn.Conv(features=4, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = nn.max_pool(x, window_shape=(4, 4), strides=(4, 4))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=8)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

class TrainerModule:
    def __init__(self,
                 model_name: str,
                 model_class: nn.Module,
                 model_hparams: dict,
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 exmp_imgs: Any,
                 seed,
                 checkpoint_path='./'):
        """
        Module for summarizing all training functionalities.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
            checkpoint_path - logging directory
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        self.model = self.model_class(**self.model_hparams)
        self.log_dir = checkpoint_path
        self.create_functions()
        self.init_model(exmp_imgs)

    def create_functions(self):
        def calculate_loss(params, batch_stats, batch, mPi, Pi_t, mode, train, lam):
            imgs, labels, metadata = batch
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    imgs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)

            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

            # Use regularizers from the Regularizers class
            l2 = Regularizers.l2_regularizer(params)
            reg, others_reg = Regularizers.laplace_regularizer(params, mPi, Pi_t, mode)

            loss_full = loss + 0 * l2 + lam * reg
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss_full, (acc, new_model_state, loss, reg, others_reg, l2)

        def train_step(state, batch, mPi, Pi_t, mode, lam):
            loss_fn = lambda params: calculate_loss(params, state.batch_stats, batch, mPi, Pi_t, mode, train=True, lam=lam)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, new_model_state, loss_part, reg, others_reg, l2 = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, loss, acc, (loss_part, reg, others_reg, l2)

        def eval_step(state, batch, mPi, Pi_t,mode, lam):
            _, (acc, _, _, _, _, _) = calculate_loss(state.params, state.batch_stats, batch, mPi, Pi_t, mode, train=False,
                                                     lam=lam)
            return acc

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        init_rng = jax.random.PRNGKey(self.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name}"'

        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales={
                int(num_steps_per_epoch * num_epochs * 0.6): 0.1,
                int(num_steps_per_epoch * num_epochs * 0.85): 0.1
            }
        )

        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))

        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **self.optimizer_hparams)
        )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
        )

    def train_model(self, train_loader, eval_loader,mode, ti, num_epochs=200, flt=None,lam=0):
        if ti == 0:
            self.init_optimizer(num_epochs, len(train_loader))
            flat_params_init, _ = jax.flatten_util.ravel_pytree(self.state.params)
            flt = Diag_LowRank(flat_params_init.size, 2)

        best_eval = 0.0
        print('lam', lam)
        for epoch_idx in tqdm(range(1, num_epochs + 1), position=0, leave=True):
            self.train_epoch(train_loader,mode, ti, epoch=epoch_idx, flt=flt, lam=lam)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(eval_loader, mode, flt, lam)
                wandb.log({"eval/acc_task_tr" + str(ti): eval_acc})
                if eval_acc >= best_eval:
                    best_eval = eval_acc
        return flt

    def train_epoch(self, train_loader,mode, ti, epoch, flt, lam):
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc='Training', position=0, leave=True):
            images, labels, metadata = batch
            self.state, loss, acc, (loss_part, reg, others_reg, l2) = self.train_step(
                self.state, batch, flt.mPi, flt.Pi_t, mode, lam=lam
            )
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
            metrics['reg'].append(reg)
            metrics['loss_no_reg'].append(loss_part)
            #metrics['l2'].append(l2)

        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            wandb.log({"train/" + key + "_task_tr" + str(ti): avg_val})

    def eval_model(self, data_loader, mode, flt=None, lam=0, extra_state=None):
        state_ = extra_state if extra_state is not None else self.state
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(state_, batch, flt.mPi, flt.Pi_t,mode, lam)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir,
            target={'params': self.state.params, 'batch_stats': self.state.batch_stats},
            step=step,
            overwrite=True
        )

    def load_model(self, pretrained=False):
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(self.log_dir, f'{self.model_name}.ckpt'),
                target=None
            )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict['params'],
            batch_stats=state_dict['batch_stats'],
            tx=self.state.tx if self.state else optax.sgd(0.1)
        )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(self.log_dir, f'{self.model_name}.ckpt'))

    def compute_GGN(self, batch, train=False):
        return AdditionalFunctions.compute_GGN(self.model, self.state, batch, train)

def train_classifier(model_name, model_class, ti, mode, trainer_p=None ,model_hparams={}, optimizer_name='adam', optimizer_hparams = {}, exmp_imgs = None, num_epochs=10, seed=0, flt=None, lam_t=0):
    # Create a trainer module with specified hyperparameters
    print(ti)
    config = GlobalRegistry.get_config()
    if config.TEST == 1:
        train_loader, _, test_loader = GlobalRegistry.get_loaders_per_task_split(ti)
        eval_loader = test_loader
    else:
        train_loader, valid_loader, _ = GlobalRegistry.get_loaders_per_task_split(ti)
        eval_loader = valid_loader
    if ti == 0:
        trainer = TrainerModule(model_name,model_class,model_hparams,optimizer_name,optimizer_hparams,exmp_imgs,seed=seed)
    else:
        trainer = trainer_p
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists

        flt = trainer.train_model(train_loader, eval_loader,mode, ti, num_epochs=num_epochs, flt=flt, lam=lam_t)
        #trainer.load_model()
    else:
        pass
        #trainer.load_model(pretrained=True)
    # Test trained model
    eval_acc = trainer.eval_model(eval_loader, mode, flt, lam_t)
    batch = next(iter(train_loader))
    J, H = trainer.compute_GGN(batch, False)

    return trainer, {'eval': eval_acc}, (J,H), flt


def training_loop_Camelyon(list_of_eval_loaders):
    config = GlobalRegistry.get_config()

    # Prepare all the logging matrices
    acc_matrix_laplace_full = np.zeros((config.T, config.T))
    avg_test_accs_laplace_full = []
    list_of_Pts = []
    batch_statss = []

    for task_id in range(config.T):

        train_loader, valid_loader, test_loader = GlobalRegistry.get_loaders_per_task_split(task_id)

        if (config.METHOD == 'ours' and task_id > 0 and config.LAMBDA == 0)\
            or (config.METHOD == 'baseline' and task_id > 0)\
            or (config.METHOD == 'ewc' and task_id > 0)\
            or (config.METHOD == 'ewc' and task_id == 0 and config.LAMBDA != 0)\
            or (config.METHOD == 'osla' and task_id > 0 and config.LAMBDA == 0):
            print('The condition was met for lambdas')
            LAM_TASK = config.LAMBDA
        elif (config.METHOD == 'baseline' and task_id == 0):
            LAM_TASK = config.LAMBDA_INIT
        else:
            LAM_TASK = 1

        images, labels, metadata = next(iter(train_loader))
        if (config.NAME_RES == 'inferred_params' and task_id %2==0) or config.NAME_RES != 'inferred_params':
            print(config.NAME_RES, 'Training', task_id, 'should not be 1 or 3')
            cnn_trainer, cnn_results, (J, H), flt = train_classifier(model_name="CNN",
                                                                 model_class=CNN,
                                                                 ti=task_id,
                                                                 mode = 0 if task_id == 0 else theta_star,
                                                                 trainer_p=None if task_id == 0 else cnn_trainer,
                                                                 model_hparams={"num_classes": 2},
                                                                 optimizer_name="adam",
                                                                 optimizer_hparams={"lr": config.LR},
                                                                 exmp_imgs=jax.device_put(images),
                                                                 num_epochs=config.NUM_EPOCHS, #if (config.NAME_RES != 'inferred_params' or (config.NAME_RES == 'inferred_params' and task_id == 0)) else 2*config.NUM_EPOCHS,
                                                                 seed=config.SEED,
                                                                 flt=None if task_id == 0 else flt,
                                                                 lam_t=LAM_TASK)

        for tii in range(task_id + 1):
            acc = cnn_trainer.eval_model(list_of_eval_loaders[tii], 0 if task_id == 0 else theta_star, flt, LAM_TASK)
            wandb.log({"eval/acc_task_ev" + str(tii): acc})
            acc_matrix_laplace_full[task_id, tii] = acc

            print('Current task:', task_id, 'Evaluated on:', tii)
            print(f'Test Accuracy (%): {acc_matrix_laplace_full[task_id, tii]:.2f}).')
        avg_test_acc_t_laplace_full = acc_matrix_laplace_full[tii, :(tii + 1)].mean()
        avg_test_accs_laplace_full.append(avg_test_acc_t_laplace_full)
        print(f'Average accuracy: {avg_test_acc_t_laplace_full:.2f}')
        if (config.NAME_RES == 'inferred_params'):
            print(jnp.diag(acc_matrix_laplace_full))
            print(np.array([acc_matrix_laplace_full[0][0], acc_matrix_laplace_full[2][1], acc_matrix_laplace_full[2][2],
                            acc_matrix_laplace_full[-1][-2], acc_matrix_laplace_full[-1][-1]]))
            print(acc_matrix_laplace_full[-1])

        LAM_TASK = config.LAMBDA#*(1 if task_id ==0 else 0)
        print('Update:', LAM_TASK)
        flt, theta_star, unflatten_func, batch_statss, Qs, thetas = kalman_step(task_id, J,H, cnn_trainer, batch_statss, list_of_Pts,flt, LAM_TASK, Qs if task_id != 0 else None, thetas if task_id != 0 else None)


    if not os.path.exists(config.DIR_RES):
        os.makedirs(config.DIR_RES)
    print(np.diag(acc_matrix_laplace_full))
    np.savetxt(os.path.join(config.DIR_RES, 'accuracies_matrix.csv'), acc_matrix_laplace_full,
               delimiter=',', comments='')
    np.savetxt(os.path.join(config.DIR_RES, 'accuracies_avr.csv'), avg_test_accs_laplace_full,
               delimiter=',', comments='')

    return theta_star, flt, thetas, list_of_Pts, Qs, batch_statss, cnn_trainer, list_of_eval_loaders, unflatten_func



