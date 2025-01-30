import yaml
import os
import argparse


class Config:
    def __init__(self, dataset_type='default'):
        '''
        Prepares the arguments for the training.
        dataset_type: str, either 'default' or 'camelyon' to determine which configuration to load
        '''
        parser = argparse.ArgumentParser(description='Lacole training')

        default_config = 'utils/configs/config_default_Camelyon.yaml'

        parser.add_argument('--config', type=str, default=default_config,
                            help='Path config file specifying model architecture and training procedure')
        args = parser.parse_args()

        with open(args.config, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)

        # Common configurations
        self.SEED = config['architecture']['seed']
        self.LAMBDA = config['training']['lambda']
        self.LAMBDA_INIT = config['training']['lambda_init']
        self.LR = config['training']['lr']
        self.TEST = config['evaluation']['test']
        self.NUM_EPOCHS = config['training']['num_epochs']
        self.BATCH_SIZE = config['training']['batch_size']
        self.DIR_SRC = config['paths']['dir_src']
        self.DIR_EXP = config['paths']['dir_exp']
        self.DIR_RES = self.DIR_SRC + self.DIR_EXP

        self._init_camelyon_config(config)

        # Create results directory if it doesn't exist
        if not os.path.exists(self.DIR_RES):
            os.makedirs(self.DIR_RES)
    def _init_camelyon_config(self, config):
        '''Initialize configuration specific to Camelyon dataset'''
        self.DATASET = 'Camelyon'
        self.METHOD = config['training']['method']
        self.TASK_NAME = config['training']['task_name']
        self.T = config['training']['num_tasks']
        self.NUM_SAMPLES = config['training']['num_samples']
        self.NAME_RES = config['training']['name_res']
        # Kernel and bias configurations
        self.Q_c0_kernel = config['training']['c0_kernel']
        self.Q_c0_bias = config['training']['c0_bias']
        self.Q_c1_kernel = config['training']['c1_kernel']
        self.Q_c1_bias = config['training']['c1_bias']
        self.Q_c2_kernel = config['training']['c2_kernel']
        self.Q_c2_bias = config['training']['c2_bias']
        self.Q_d0_kernel = config['training']['d0_kernel']
        self.Q_d0_bias = config['training']['d0_bias']
        self.Q_d1_kernel = config['training']['d1_kernel']
        self.Q_d1_bias = config['training']['d1_bias']
        self.Q_ALL = config['training']['Q_all']


    def save_to_yaml(self, filename="config_saved.yaml"):
        '''Save current configuration to a YAML file'''
        config_dict = {key: getattr(self, key) for key in dir(self)
                       if not key.startswith('__') and not callable(getattr(self, key))}

        # Remove dynamic/runtime attributes
        config_dict.pop('DIR_RES', None)
        config_dict.pop('main_rng', None)

        with open(os.path.join(self.DIR_RES, filename), 'w') as setup_save:
            yaml.dump(config_dict, setup_save, default_flow_style=False)