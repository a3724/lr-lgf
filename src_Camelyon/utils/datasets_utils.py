from typing import Iterator
from typing import Any, Callable, Optional
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random
import os
import gzip
import tarfile
import zipfile
import pandas as pd
import tqdm
from .base_utils import *

# Helper functions from https://github.com/p-lambda/wilds

def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)

def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def gen_bar_updater(total) -> Callable[[int, int, int], None]:
    pbar = tqdm(total=total, unit='Byte')

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def download_url(url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, size: Optional[int] = None) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(size)
            )
        except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(size)
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")

def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")

def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")

def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")

def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")

def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")

def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")

def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

# End of helper function, below adjusted Camelyon dataset loader

class CamelyonDataset():

    def __init__(self, version='1.0', root_dir='data', download=False, perc=0, which=0, task_name='percentages'):
        self._version = version
        self._dataset_name = 'camelyon17'
        print(os.getcwd())
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96, 96)
        self._perc = perc
        self.which = which
        self.task_name = task_name

        # Read in metadata
        print('self._data_dir', self._data_dir)
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        # Get the y values
        self._y_array = self._metadata_df['tumor'].values
        self._y_size = 1
        self._n_classes = 2

        self._split_dict = {
            'valid': 3,
            'train': 2,
            'test': 1,
            'not_used': 0
        }
        self._split_names = {
            'valid': 'Valid',
            'train': 'Train',
            'test': 'Test',
            'not_used': 'NotUsed'
        }
        config = GlobalRegistry.get_config()

        self.config = config
        # Extract splits
        centers = self._metadata_df['center'].values
        num_centers = int(np.max(centers)) + 1

        # Initialize split column
        self._metadata_df['split'] = self._split_dict['not_used']

        # Assign train and test splits for group 0_1_2_3
        # num_samples = np.sum(mask)
        num_samples = self.config.NUM_SAMPLES

        if self.task_name == 'percentages':
            center_mask = self._metadata_df['center'] == 0
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
            center_mask = self._metadata_df['center'] == 1
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=False)
        elif self.task_name == 'new_labels':
            center_mask = self._metadata_df['center'] < 10
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
        elif self.task_name == 'corners':
            center_mask = self._metadata_df['center'] == 1
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
        elif self.task_name == 'gradual_lighting':
            center_mask = self._metadata_df['center'] < 1
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
        elif self.task_name == 'imbalanced':
            center_mask = self._metadata_df['center'] == 0
            num_samples = sum(center_mask)
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=True)
            center_mask = self._metadata_df['center'] == 1
            num_samples = sum(center_mask)
            self._assign_split(center_mask, num_samples, perc=self._perc, main_group=False)




        # Update metadata array
        self._split_array = self._metadata_df['split'].values
        self._metadata_array = np.stack((centers, self._metadata_df['slide'].values, self._y_array), axis=1)
        self._metadata_fields = ['hospital', 'slide', 'y']

        filtered_df = self._metadata_df[self._metadata_df['split'] == 2].sample(frac=1)
        file_paths_train = [
            os.path.join(self._data_dir,
                         f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png')
            for patient, node, x, y in filtered_df[['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False,
                                                                                                         name=None)
        ]
        labels_train = list(filtered_df['tumor'].values)
        metadata_list_train = list(filtered_df['center'].values)

        filtered_df = self._metadata_df[self._metadata_df['split'] == 1]
        file_paths_test = [
            os.path.join(self._data_dir,
                         f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png')
            for patient, node, x, y in filtered_df[['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False,
                                                                                                         name=None)
        ]
        labels_test = list(filtered_df['tumor'].values)
        metadata_list_test = list(filtered_df['center'].values)

        filtered_df = self._metadata_df[self._metadata_df['split'] == 3]
        file_paths_valid = [
            os.path.join(self._data_dir,
                         f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png')
            for patient, node, x, y in filtered_df[['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False,
                                                                                                         name=None)
        ]
        labels_valid = list(filtered_df['tumor'].values)
        metadata_list_valid = list(filtered_df['center'].values)

        dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train, metadata_list_train))
        dataset_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test, metadata_list_test))
        dataset_valid = tf.data.Dataset.from_tensor_slices((file_paths_valid, labels_valid, metadata_list_valid))


        def parse_function(file_path, label, metadata):
            # Read and decode the image file
            image = tf.io.read_file(file_path)
            image = tf.image.decode_png(image, channels=3)  # Adjust channels based on your image format
            image = image / 255
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


            if self.task_name == 'new_labels':
                label = label + int(5*self.which)

            elif self.task_name == 'corners':
                if self.which == 0:
                    # Crop the image to the top-left corner (10x10)
                    cropped_image = image[:5, :5, :]
                elif self.which == 1:
                    # Crop the image to the top-right corner (10x10)
                    cropped_image = image[:5, -5:, :]
                elif self.which == 2:
                    # Crop the image to the bottom-left corner (10x10)
                    cropped_image = image[-5:, :5, :]
                elif self.which == 3:
                    # Crop the image to the bottom-right corner (10x10)
                    cropped_image = image[-5:, -5:, :]
                else:
                    raise ValueError("Invalid value for which. It should be between 0 and 3.")

                    # Get the shape of the cropped image
                cropped_height = tf.shape(cropped_image)[0]
                cropped_width = tf.shape(cropped_image)[1]

                # Pad the cropped image with zeros to maintain the original shape
                image = tf.pad(cropped_image, paddings=[[0, tf.maximum(0, tf.shape(image)[0] - cropped_height)],
                                                               [0, tf.maximum(0, tf.shape(image)[1] - cropped_width)],
                                                               [0, 0]])

            elif self.task_name == 'gradual_lighting':
                _SCALE = 0.6
                _GRADUAL_TASKS = 4 if config.NAME_RES != 'inferred_params' else 8
                _GRADUAL_SCALE = 0.1 if config.NAME_RES != 'inferred_params' else 0.05
                print(_SCALE * (-(_GRADUAL_TASKS - self.which) * _GRADUAL_SCALE + _GRADUAL_SCALE * self.which))
                image = tf.image.adjust_brightness(image, delta=_SCALE*(-(_GRADUAL_TASKS-self.which)*_GRADUAL_SCALE + _GRADUAL_SCALE*self.which))
                image = tf.clip_by_value(image, 0.0, 1.0)

            image = (image - _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN) / _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD
            return image, label, metadata

        dataset_train = dataset_train.map(parse_function)
        dataset_test = dataset_test.map(parse_function)
        dataset_valid = dataset_valid.map(parse_function)
        # dataset = dataset.shuffle(buffer_size=len(file_paths))

        # Batch the datasets
        batch_size = self.config.BATCH_SIZE
        self.dataset_train = dataset_train.batch(batch_size)
        self.dataset_test = dataset_test.batch(batch_size)
        self.dataset_valid = dataset_valid.batch(batch_size)

    def initialize_data_dir(self, root_dir, download):
        os.makedirs(root_dir, exist_ok=True)
        data_dir = os.path.join(root_dir, f'{self._dataset_name}_v{self._version}')
        version_file = os.path.join(data_dir, f'RELEASE_v{self._version}.txt')
        _versions_dict = {
            '1.0': {
                'download_url': 'https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/',
                'compressed_size': 10_658_709_504}}


        # If the dataset exists at root_dir, then don't download.
        if not (os.path.exists(data_dir) and os.path.exists(version_file)):
            print(os.path.exists(data_dir), data_dir, os.getcwd())
            version_dict = _versions_dict[self._version]
            download_url_ = version_dict['download_url']
            compressed_size = version_dict['compressed_size']
            print(f'Downloading dataset to {data_dir}...')
            download_root = os.path.expanduser(data_dir)
            extract_root = download_root
            download_url(download_url_, download_root, 'archive.tar.gz', None, compressed_size)

            archive = os.path.join(download_root, 'archive.tar.gz')
            print("Extracting {} to {}".format(archive, extract_root))
            extract_archive(archive, extract_root, True)
        print('all good')

        return data_dir

    def _assign_split(self, mask, num_samples, perc, main_group):
        # Determine the number of samples for train and test in the masked subset

        num_train = int(num_samples)
        num_test = 2000 #max(int(0.5*(num_samples - num_train)), 1000)
        num_valid = 2000 #max(num_samples - num_train - num_test, 1000)

        # Adjust train/test ratio based on frac and group type
        if main_group:
            num_train_main = int(num_train * (1 - perc))
            num_test_main = int(num_test * (1 - perc))
            num_valid_main = int(num_valid * (1 - perc))
        else:
            num_train_main = int(num_train * perc)
            num_test_main = int(num_test * perc)
            num_valid_main = int(num_valid * perc)

        # Randomly assign samples to train or test
        indices = np.where(mask)[0]
        np.random.shuffle(indices)
        train_indices = indices[:num_train_main]
        test_indices = indices[num_train_main:num_train_main + num_test_main]
        valid_indices = indices[num_train_main + num_test_main:num_train_main + num_test_main + num_valid_main]

        # Update the split values in the dataframe
        self._metadata_df.loc[train_indices, 'split'] = self._split_dict['train']
        self._metadata_df.loc[test_indices, 'split'] = self._split_dict['test']
        self._metadata_df.loc[valid_indices, 'split'] = self._split_dict['valid']


def load_all_splits():
    config = GlobalRegistry.get_config()
    percentages = [0,0,0,0]
    list_of_eval_loaders = []
    for task_id in range(config.T):
        print('Preparing...')

        full_dataset = CamelyonDataset(root_dir='src_Camelyon/data', perc=percentages[0], which=task_id, task_name=config.TASK_NAME)
        GlobalRegistry.set_camelyon_per_task(task_id, full_dataset)

        train_loader = tfds.as_numpy(full_dataset.dataset_train)
        valid_loader = tfds.as_numpy(full_dataset.dataset_valid)
        test_loader = tfds.as_numpy(full_dataset.dataset_test)
        GlobalRegistry.set_loaders_per_task_split(task_id, train_loader, valid_loader, test_loader)
        if config.TEST == 1:
            list_of_eval_loaders.append(test_loader)
        else:
            list_of_eval_loaders.append(valid_loader)

    return list_of_eval_loaders