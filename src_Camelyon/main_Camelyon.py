
try:
    import wandb
except Exception as e:
    pass
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0


XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
TF_DETERMINISTIC_OPS=1
TF_CUDNN_DETERMINISTIC=1




from utils.DiagLowRank import *
from utils.datasets_utils import *
from utils.train_utils import *
from utils.args_utils import *
from utils.regularizer_utils import *
from utils.base_utils import GlobalRegistry

# Parse input arguments
config = Config('camelyon')
GlobalRegistry.set_config(config)
config.save_to_yaml("config_saved.yaml")

# Seeds.
key = jax.random.PRNGKey(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

# Logging.
wandb.init(
    project=str(config.METHOD)+'_'+str(config.TASK_NAME),
    config=config.__dict__)

# Load datasets
list_of_test_loaders = load_all_splits()

# Training loop
theta_star, flt, thetas, list_of_Pts, Qs, batch_statss, cnn_trainer, list_of_eval_loaders, unflatten_func = training_loop_Camelyon(list_of_test_loaders)

# Smoother
ms_smoother = smoother_run(theta_star, flt, thetas, list_of_Pts, Qs, batch_statss, cnn_trainer, list_of_eval_loaders, unflatten_func)
print(ms_smoother)
np.savetxt(os.path.join(config.DIR_RES, 'accuracies_smoother.csv'), ms_smoother,
               delimiter=',', comments='')
