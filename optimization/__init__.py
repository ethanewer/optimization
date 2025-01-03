from .lbfgs import fmin_lbfgs

__all__ = [
    "fmin_lbfgs",
    "functionalize_torch",
    "fmin_lbfgs_torch",
    "functionalize_mlx",
    "mlx_to_numpy",
]


def functionalize_torch(*args, **kwargs):
    from .torch_optimization import functionalize_torch as _functionalize_torch

    return _functionalize_torch(*args, **kwargs)


def fmin_lbfgs_torch(*args, **kwargs):
    from .torch_optimization import fmin_lbfgs_torch as _fmin_lbfgs_torch

    return _fmin_lbfgs_torch(*args, **kwargs)


def functionalize_mlx(*args, **kwargs):
    from .mlx_util import functionalize_mlx as _functionalize_mlx

    return _functionalize_mlx(*args, **kwargs)


def mlx_to_numpy(*args, **kwargs):
    from .mlx_util import mlx_to_numpy as _mlx_to_numpy

    return _mlx_to_numpy(*args, **kwargs)
