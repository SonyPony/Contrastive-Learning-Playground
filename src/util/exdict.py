import operator
import hydra
import copy

from collections.abc import MutableMapping
from omegaconf import DictConfig, OmegaConf, SCMode
from functools import reduce


class ExDict(MutableMapping):
    """Implementation a dictionary with deepcopy and copy methods."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        v = self._data[key]
        if isinstance(v, dict):
            return ExDict(v)
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    def __copy__(self):
        return ExDict(copy.copy(self._data))

    def __deepcopy__(self, memo):
        return ExDict(copy.deepcopy(self._data))

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__


class ExperimentLoader:
    """ExperimentLoader resolves the yaml hydra config."""

    @staticmethod
    def load_data(cfg: DictConfig):
        # register sum and product algebraic operators
        operations = {
            "product": lambda x: reduce(operator.mul, x, 1),
            "sum": sum
        }

        OmegaConf.register_new_resolver(
            "op", lambda op, *params: operations[op](params)
        )

        # resolve and instantiate objects within config
        OmegaConf.resolve(cfg)
        cfg = hydra.utils.instantiate(cfg, _recursive_=True)

        # return cfg as simple containers
        return ExDict(
            OmegaConf.to_container(
                cfg=cfg,
                throw_on_missing=True,
                structured_config_mode=SCMode.INSTANTIATE
            )
        )