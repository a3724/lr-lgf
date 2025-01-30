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
from .base_utils import Batch


def load_dataset(split: str, *,
                 shuffle: bool = False,
                 batch_size: int = 32,
                 input_dim: int = 28,
                 seed: int = 0,
                 task: int = 0,
                 downsample: bool = False,
                 permute: bool = False,
                 disjoint_split: bool = False,
                 label_split: Optional[int] = None,
                 normalize: str = 'minmax'):
    """
    Loads and preprocesses the MNIST dataset according to specified transformations.

    This function allows for flexible configurations on the MNIST dataset, including pixel
    permutation, downsampling, label filtering, and task-based disjoint splits.

    Parameters:
        split (str): Dataset split to load ("train" or "test").
        shuffle (bool): If True, shuffles the dataset.
        batch_size (int): Number of samples per batch.
        input_dim (int): Target dimension for the image (used if downsampling or permuting).
        seed (int): Seed for reproducibility in dataset shuffling and permutations.
        task (int): Task identifier; affects permutations and label filtering.
        downsample (bool): If True, downsamples images by a factor based on `input_dim`.
        permute (bool): If True, permutes pixel positions based on `task`.
        disjoint_split (bool): If True, splits dataset into two disjoint sets based on `task`.
        label_split (Optional[int]): If provided, restricts labels to two random classes.
        normalize (str): Normalization method. Options:
            - 'minmax': Scale to [0, 1] range (default)
            - 'zscore': Zero-mean, unit variance
            - None: No normalization

    Returns:
        Iterator[Batch]: An iterator over batches of the processed dataset.

    Usage:
        - To load unaltered MNIST: set all options to False.
        - To load MNIST with permuted pixels: set `permute=True`.
        - To load downsampled MNIST: set `downsample=True`.
        - To filter the dataset to specific labels (split by task): set `label_split=n`.
        - To obtain disjoint label splits (0-4 and 5-9 based on task): set `disjoint_split=True`.
    """

    np.random.seed(seed + task)

    # Set up permutation if requested
    if permute:
        perm_inds = np.arange(input_dim * input_dim)
        if task > 0:  # Only permute for tasks > 0
            np.random.shuffle(perm_inds)

    # Set up disjoint or label-based splits
    if disjoint_split:
        chosen_idx = tf.constant([0, 1, 2, 3, 4] if task == 0 else [5, 6, 7, 8, 9], dtype=tf.int64)
    elif label_split is not None:
        random.seed(seed + task)
        chosen_idx = tf.constant(random.sample(range(10), label_split), dtype=tf.int64)

    def downsample_image(x):
        """Downsamples the image if downsample is enabled."""
        scale_factor = int(np.round(28 / input_dim))
        x["image"] = x["image"][::scale_factor, ::scale_factor]
        return x

    def permute_image(x):
        """Permutes the image pixels if permute is enabled."""
        im = tf.reshape(x["image"], [-1])
        im = tf.gather(im, perm_inds)
        x["image"] = tf.reshape(im, (input_dim, input_dim, 1))
        return x

    def normalize_image(x):
        """Normalizes the image based on specified method."""
        image = x["image"]

        if normalize == 'minmax':
            # Scale to [0, 1] range
            image = tf.cast(image, tf.float32) / 255.0
        elif normalize == 'zscore':
            # Zero-mean, unit variance normalization
            image = tf.cast(image, tf.float32)
            mean = tf.reduce_mean(image)
            std = tf.math.reduce_std(image)
            image = (image - mean) / (std + 1e-7)  # Add small epsilon to avoid division by zero

        x["image"] = image
        return x

    def filter_labels(x):
        """Filters the dataset based on chosen label indices."""
        condition = tf.reduce_any(tf.equal(chosen_idx, x["label"]))
        placeholder = {"image": tf.zeros_like(x["image"]), "label": tf.constant(-1, dtype=tf.int64)}
        return tf.cond(condition, lambda: x, lambda: placeholder)

    # Load dataset
    ds, ds_info = tfds.load("mnist:3.*.*", split=split, with_info=True)
    ds = ds.cache()

    # Apply transformations conditionally
    if downsample:
        ds = ds.map(downsample_image)

    if normalize:
        ds = ds.map(normalize_image)

    if permute:
        ds = ds.map(permute_image)
    if disjoint_split or label_split is not None:
        ds = ds.map(filter_labels).filter(lambda x: tf.not_equal(x["label"], -1))

    if shuffle:
        ds = ds.shuffle(ds_info.splits[split].num_examples, seed=seed)

    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: Batch(**x))

    return iter(tfds.as_numpy(ds))


def generate_splits(config):
    '''
    Generates the training, evaluation, and validation datasets for the given configuration.
    '''
    normalize = None
    if config.DATASET == 'MNIST_permuted':
        permute = True
        disjoint_split = False
        label_split = None
    elif config.DATASET == 'MNIST_split':
        permute = False
        disjoint_split = False
        label_split = 2
    elif config.DATASET == 'MNIST_disjoint':
        permute = False
        disjoint_split = True
        label_split = None

    train_datasets = [
        load_dataset('train', shuffle=True, batch_size=config.BATCH_SIZE, input_dim=config.INPUT_IMAGE_SIZE, seed=0,
                                        task=task_data, permute=permute, disjoint_split=disjoint_split, label_split=label_split, normalize=normalize)
        for task_data in range(config.T)]
    eval_datasets = [
        load_dataset('test[:50%]', shuffle=False, batch_size=config.BATCH_SIZE, input_dim=config.INPUT_IMAGE_SIZE, seed=0,
                                        task=task_data, permute=permute, disjoint_split=disjoint_split,label_split=label_split, normalize=normalize)
        for task_data in range(config.T)]

    valid_datasets = [
        load_dataset('test[50%:]', shuffle=False, batch_size=config.BATCH_SIZE, input_dim=config.INPUT_IMAGE_SIZE, seed=0,
                    task=task_data, permute=permute,disjoint_split=disjoint_split, label_split=label_split, normalize=normalize)
        for task_data in range(config.T)]

    if config.BATCH_SIZE_HES != None:
        train_datasets_hes = [
            load_dataset('train', shuffle=True, batch_size=config.BATCH_SIZE_HES, input_dim=config.INPUT_IMAGE_SIZE, seed=0,
                        task=task_data, permute=permute,disjoint_split=disjoint_split, label_split=label_split, normalize=normalize)
            for task_data in range(config.T)]
    else: train_datasets_hes = None

    return train_datasets, eval_datasets, valid_datasets, train_datasets_hes

