from typing import Any, Callable

import mlx.core as mx
import numpy as np
from mlx import nn, utils


def functionalize_mlx(module: nn.Module) -> tuple[Callable, mx.array]:
    names, params = zip(*utils.tree_flatten(module.state.parameters()))
    shapes = [p.shape for p in params]
    split_idxs = np.cumsum([np.prod(s) for s in shapes[:-1]])
    params = mx.concatenate([p.reshape(-1) for p in params])

    def call_module(params: mx.array, *args) -> Any:
        split_params = mx.split(params, split_idxs)
        reshaped_params = [p.reshape(s) for p, s in zip(split_params, shapes)]
        tree = utils.tree_unflatten(list(zip(names, reshaped_params)))
        return module.update(tree)(*args)

    return call_module, params


def mlx_to_numpy(f: Callable) -> Callable:
    def callback(x, *args, **kwargs):
        out = f(mx.array(x), *args, **kwargs)
        mx.eval(out)
        if isinstance(out, tuple):
            return tuple(np.array(y, copy=False) for y in out)
        else:
            return np.array(out, copy=False)

    return callback
