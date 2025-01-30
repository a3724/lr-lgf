import yaml
import os
import argparse

class Config:

    def __init__(self):
        '''
        Prepares the arguments for the training.
        '''

        parser = argparse.ArgumentParser(description='Lacole training')
        parser.add_argument('--config', type=str, default='utils/configs/config_default.yaml',
                            help='Path config file specifying model '
                                'architecture and training procedure')
        args = parser.parse_args()

        with open(args.config, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)

        self.METHOD = config['training']['method']
        self.DATASET = config['data']['dataset']

        if self.DATASET == 'MNIST_permuted':
            self.NUM_DATA = 60000
        elif self.DATASET == 'MNIST_split':
            self.NUM_DATA = int(60000/5)
        elif self.DATASET == 'MNIST_disjoint':
            self.NUM_DATA = int(60000/2)

        self.BATCH_SIZE = config['training']['batch_size']
        self.BATCH_SIZE_HES = config['training'].get('batch_size_hes', None)
        self.INPUT_IMAGE_SIZE = config['data']['input_image_size']
        self.LEARNING_RATE = config['training']['lr']
        self.WEIGHT_DECAY = config['training']['weight_decay']
        self.NUM_MID = config['architecture']['num_mid']
        self.NUM_LAYERS = config['architecture']['num_l']
        self.NUM_CLASSES = config['architecture']['num_classes']
        self.TEST = config['evaluation']['test']


        if self.METHOD == 'ours':
            self.k = config['training']['rank']

            self.ALPHA_LOW_W = config['training']['alpha_low_w']
            self.ALPHA_MID_W = config['training']['alpha_mid_w']
            self.ALPHA_HIGH_W = config['training']['alpha_high_w']

            self.ALPHA_LOW_B = config['training']['alpha_low_b']
            self.ALPHA_MID_B = config['training']['alpha_mid_b']
            self.ALPHA_HIGH_B = config['training']['alpha_high_b']

        self.LAMBDA_INIT = config['training']['lambda_init']
        self.LAMBDA = config['training']['lambda']
        self.IN_DIM, self.MID_DIM, self.OUT_DIM = self.INPUT_IMAGE_SIZE * self.INPUT_IMAGE_SIZE, self.NUM_MID, self.NUM_CLASSES
        self.ONE_DATASET_RUN = int(self.NUM_DATA / self.BATCH_SIZE)
        self.NUM_EPOCHS = config['training']['num_epochs']
        self.EPOCH = self.NUM_EPOCHS * self.ONE_DATASET_RUN
        self.HOW_MANY_EVALS = self.ONE_DATASET_RUN
        self.T = config['training']['num_tasks']
        self.SEED = config['architecture']['seed']

        self.DIR_SRC = config['paths']['dir_src']
        self.DIR_EXP = config['paths']['dir_exp']
        self.DIR_RES = self.DIR_SRC+self.DIR_EXP
        self.SAVE_PI_T = config['training']['save_pi_t']

        if not os.path.exists(self.DIR_RES):
            os.makedirs(self.DIR_RES)

    def save_to_yaml(self, filename="config_saved.yaml"):
        # Convert all class attributes to a dictionary
        config_dict = {key: getattr(self, key) for key in dir(self) if not key.startswith('__')}

        # Remove 'DIR_RES' because it's a path generated dynamically
        config_dict.pop('DIR_RES', None)

        # Write the dictionary to a YAML file
        with open(os.path.join(self.DIR_RES, filename), 'w') as setup_save:
            yaml.dump(config_dict, setup_save, default_flow_style=False)