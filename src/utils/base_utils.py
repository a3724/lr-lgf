import haiku as hk
from typing import NamedTuple
import numpy as np
import jax
import optax


class GlobalRegistry:
    _network = None
    _net_batched = None
    _loss_ = None
    _optimizer = None
    _loss_batched = None
    _loss_no_reg_batched = None
    _compute_GGN_batched = None
    _compute_diag_fim_batched = None
    _config = None

    @classmethod
    def set_network(cls, network):
        cls._network = network

    @classmethod
    def set_net_batched(cls, net_batched):
        cls._net_batched = net_batched

    @classmethod
    def set_optimizer(cls, optimizer):
        cls._optimizer = optimizer

    @classmethod
    def set_config(cls, config):
        cls._config = config

    @classmethod
    def set_loss_batched(cls, _loss_batched):
        cls._loss_batched = _loss_batched

    @classmethod
    def set_loss_(cls, loss_):
        cls._loss_ = loss_

    @classmethod
    def get_network(cls):
        return cls._network

    @classmethod
    def get_net_batched(cls):
        return cls._net_batched

    @classmethod
    def get_optimizer(cls):
        return cls._optimizer

    @classmethod
    def get_loss_batched(cls):
        return cls._loss_batched

    @classmethod
    def get_loss_(cls):
        return cls._loss_

    @classmethod
    def get_loss_no_reg_batched(cls):
        return cls._loss_no_reg_batched

    @classmethod
    def get_compute_GGN_batched(cls):
        return cls._compute_GGN_batched

    @classmethod
    def get_compute_diag_fim_batched(cls):
        return cls._compute_diag_fim_batched

    @classmethod
    def get_config(cls):
        return cls._config

    @classmethod
    def set_net_batched(cls, net_batched):
        cls._net_batched = net_batched

    # Placeholder methods to be set during initialization
    @classmethod
    def set_compute_GGN_batched(cls, _compute_GGN):
        cls._compute_GGN_batched = _compute_GGN

    @classmethod
    def set_compute_diag_fim_batched(cls, _compute_diag_fim):
        cls._compute_diag_fim_batched = _compute_diag_fim

    @classmethod
    def set_loss_no_reg_batched(cls, loss_fn):
        cls._loss_no_reg_batched = loss_fn



class Batch(NamedTuple):
    """
    Represents a batch of data with images and corresponding labels.
    """
    image: np.ndarray
    label: np.ndarray