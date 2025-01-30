import haiku as hk
from typing import NamedTuple
import numpy as np
import jax
import optax

from flax.training import train_state
from typing import NamedTuple, Any


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any


class GlobalRegistry:
    _config = None
    _cam = {}
    _loaders = {}

    @classmethod
    def set_config(cls, config):
        cls._config = config

    @classmethod
    def set_camelyon_per_task(cls, task_id, cam):
        if task_id not in cls._cam:
            cls._cam[task_id] = {}
        cls._cam[task_id]['full'] = cam

    @classmethod
    def set_loaders_per_task_split(cls, task_id, train_loader, valid_loader, test_loader):
        if task_id not in cls._loaders:
            cls._loaders[task_id] = {}
        cls._loaders[task_id].update({
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader
        })

    @classmethod
    def get_config(cls):
        return cls._config

    @classmethod
    def get_camelyon_per_task(cls):
        return cls._cam

    @classmethod
    def get_loaders_per_task_split(cls, task_id):
        loaders = cls._loaders.get(task_id, {})
        return (
            loaders.get('train'),
            loaders.get('valid'),
            loaders.get('test')
        )

