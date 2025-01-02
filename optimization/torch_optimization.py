from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call  # type: ignore


def functionalize_torch(module: nn.Module) -> tuple[Callable[..., Any], Tensor]:
    state_keys = module.state_dict().keys()
    shapes = [p.shape for p in module.parameters()]
    split_size = [torch.numel(p) for p in module.parameters()]
    params = torch.cat([p.flatten() for p in module.parameters()]).detach()

    def call_module(params: Tensor, *args) -> Any:
        split_params = torch.split(params, split_size)
        reshaped_params = [p.view(s) for p, s in zip(split_params, shapes)]
        return functional_call(module, dict(zip(state_keys, reshaped_params)), *args)

    return call_module, params


def directional_value_and_grad(
    f: Callable[..., Tensor],
    x: Tensor,
    d: Tensor,
    t: Tensor | float,
    args: Sequence[Any],
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        x += t * d

    l = f(x, *args)
    l.backward()
    l.detach_()
    assert x.grad is not None
    g = x.grad.detach().clone()
    x.grad.zero_()
    with torch.no_grad():
        x -= t * d

    return l, g


def backtracking_line_search(
    f: Callable[..., Tensor],
    x: Tensor,
    l0: Tensor,
    g0: Tensor,
    d: Tensor,
    alpha: float,
    beta: float,
    max_iters: int,
    args: Sequence[Any],
) -> tuple[float, Tensor, Tensor]:
    t = 1.0
    l, g = directional_value_and_grad(f, x, d, t, args)
    for _ in range(max_iters):
        if l <= l0 + alpha * t * g0 @ d:
            break

        t *= beta
        l, g = directional_value_and_grad(f, x, d, t, args)

    return t, l, g


def fmin_lbfgs_torch(
    f: Callable[..., Tensor],
    x: Tensor,
    args: Sequence[Any] = (),
    history_size: int = 10,
    tol: Optional[float] = None,
    max_iters: int = 25,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_ls_iters: int = 10,
) -> Tensor:
    s = torch.ones((history_size, len(x)), dtype=x.dtype, device=x.device)
    y = torch.ones((history_size, len(x)), dtype=x.dtype, device=x.device)
    rho = torch.ones(history_size, dtype=x.dtype, device=x.device)
    a = torch.ones(history_size, dtype=x.dtype, device=x.device)

    x.requires_grad = True
    l = f(x, *args)
    l.backward()
    l.detach_()
    assert x.grad is not None
    g = x.grad.detach().clone()
    x.grad.zero_()

    for k in range(max_iters):
        q = g.clone()
        for i in range(k - 1, max(k - history_size - 1, -1), -1):
            j = i % history_size
            a[j] = rho[j] * s[j] @ q
            q -= a[j] * y[j]

        j = (k - 1) % history_size
        gamma = (s[j] @ y[j]) / (y[j] @ y[j])
        z = gamma * q

        for i in range(max(k - history_size, 0), k):
            j = i % history_size
            b = rho[j] * y[j] @ z
            z += s[j] * (a[j] - b)

        t, l, g_new = backtracking_line_search(  # type: ignore
            f, x, l, g, -z, alpha, beta, max_ls_iters, args
        )

        with torch.no_grad():
            x -= t * z

        if tol is not None:
            if torch.linalg.norm(g_new) < tol:
                break

        if k < max_iters - 1:
            j = k % history_size
            s[j] = -t * z
            y[j] = g_new - g
            rho[j] = 1 / (y[j] @ s[j])
            g = g_new

    return x
