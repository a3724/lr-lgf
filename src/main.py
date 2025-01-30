from utils.DiagLowRank import *
from utils.base_utils import GlobalRegistry, Batch
from utils.datasets_utils import *
from utils.train_utils import *
from utils.args_utils import *
from utils.regularizer_utils import *

import tqdm
from matplotlib import pyplot as plt
from tueplots import bundles
import os
import yaml
import argparse
import wandb


plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi": 200})

# Parse input arguments
config = Config()
GlobalRegistry.set_config(config)
config.save_to_yaml("config_saved.yaml")


# Seeds.
key = jax.random.PRNGKey(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

# Logging.
wandb.init(
    project=str(config.METHOD),
    config=config.__dict__
)

# First, make the network and optimizer.
if config.METHOD == 'osla':
    create_dynamic_mlp_with_module = hk.multi_transform(create_dynamic_mlp_with_module)
    intermediate_transformations = create_dynamic_mlp_with_module.apply
else:
    network = hk.without_apply_rng(hk.transform(net_fn))
    GlobalRegistry.set_network(network)

optimizer = optimizer_fn(weight_decay=True, weight_decay_val=config.WEIGHT_DECAY, lr=config.LEARNING_RATE)
GlobalRegistry.set_optimizer(optimizer)
# Set matrices.
acc_matrix = np.zeros((config.T, config.T))
avg_test_accs = []

# Datasets
train_datasets, eval_datasets, valid_datasets, train_datasets_hes = generate_splits(config)


# Initialize network and optimizer; note we draw an input to get shapes.
if config.METHOD == 'osla':
    initial_params = create_dynamic_mlp_with_module.init(key, jnp.zeros((config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE, 1)))
else:
    initial_params = network.init(jax.random.PRNGKey(seed=config.SEED), jnp.zeros((config.INPUT_IMAGE_SIZE, config.INPUT_IMAGE_SIZE, 1)))
initial_opt_state = optimizer.init(initial_params)
state = TrainingState(initial_params, initial_opt_state)

state_before = state
flat_params_init, _ = jax.flatten_util.ravel_pytree(initial_params)


# Prepare batched
if config.METHOD == 'osla':
    loss_batched = loss_net_xL_
    net_batched = [jax.vmap(fn, in_axes=(None, None, 0)) for fn in intermediate_transformations]
else:
    loss_batched = jax.vmap(loss_, in_axes=(None, 0))
    GlobalRegistry.set_loss_(loss_)
    net_batched = jax.vmap(network.apply, in_axes=(None, 0))
GlobalRegistry.set_loss_batched(loss_batched)
GlobalRegistry.set_net_batched(net_batched)
if config.METHOD == 'ours':
    loss_no_reg_batched = jax.vmap(loss_no_reg_, in_axes=(0, 0))
    compute_GGN_batched = jax.vmap(compute_GGN, in_axes=(None, 0))
    GlobalRegistry.set_loss_no_reg_batched(loss_no_reg_batched)
    GlobalRegistry.set_compute_GGN_batched(compute_GGN_batched)
elif config.METHOD == 'ewc':
    compute_diag_fim_batched = jax.vmap(compute_diag_fim, in_axes=(None,0))
    GlobalRegistry.set_compute_diag_fim_batched(compute_diag_fim_batched)

if config.METHOD == 'ours':
    prior_mean = np.zeros_like(flat_params_init)
elif config.METHOD == 'ewc':
    prior_mean = np.zeros((config.T+1,flat_params_init.shape[0]))
    prior_prec = np.zeros((config.T + 1, flat_params_init.shape[0]))
    reg_coef = (config.LAMBDA_INIT/config.LAMBDA) if config.LAMBDA != 0 else config.LAMBDA_INIT
    prior_prec[0,:] = reg_coef*np.ones((1,flat_params_init.shape[0]))
elif config.METHOD == 'baseline':
    prior_mean = np.zeros_like(flat_params_init)
elif config.METHOD == 'osla':
    flat_params_init, _ = jax.flatten_util.ravel_pytree(initial_params)
    initial_params_with_zeros = jax.tree_util.tree_map(jnp.zeros_like, initial_params)
    prior_mean = initial_params_with_zeros
    _, list_a, list_g = update_lists(state.params, next(train_datasets_hes[0]))
    new_KFAC = compute_KFAC_matrix(len(list_a), list_a, list_g)
    new_KFAC = [[jnp.diag(jnp.diag(config.LAMBDA_INIT*jnp.ones_like(i[0]))), jnp.diag(jnp.diag(jnp.ones_like(i[1])))] for i in new_KFAC]
    prior_prec = new_KFAC
else:
    raise 'Error'

if config.METHOD == 'ours':
    flt = Diag_LowRank(prior_mean.size, int(config.k*config.NUM_CLASSES))
    list_of_states = []


### TRAINING LOOP
state, acc_matrix, avg_test_accs = training_loop(state,
                                                 train_datasets, train_datasets_hes, valid_datasets if config.TEST == 0 else eval_datasets,  acc_matrix, avg_test_accs, flat_params_init,
                                                 flt if config.METHOD == 'ours' else None, prior_prec if (config.METHOD == 'ewc' or config.METHOD =='osla') else None, prior_mean,
                                                 config.SAVE_PI_T)


# Save accuracies
np.savetxt(os.path.join(config.DIR_RES, 'accuracies_matrix.csv'), acc_matrix,
                       delimiter=',', comments='')
np.savetxt(os.path.join(config.DIR_RES, 'accuracies_avr.csv'), avg_test_accs,
                       delimiter=',', comments='')
