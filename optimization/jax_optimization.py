from typing import Any, Callable, Sequence

import jax.numpy as jnp
from jax import Array, grad, jit
from jax.lax import scan


def lbfgs(
    f: Callable[..., Array],
    w0: Array,
    args: Sequence[Any] = (),
    m: int = 10,
    lr: float = 1.0,
    n_iters: int = 25,
) -> Array:
    n = len(w0)
    grad_f = jit(grad(f))

    def lbfgs_step(w, g, s, y, rho):
        q = g.copy()
        states = jnp.concatenate((s, y, rho), axis=1)

        def loop1(q, state):
            s = state[:n]
            y = state[n : 2 * n]
            rho = state[2 * n]
            alpha = rho * s @ q
            q -= alpha * y
            return q, alpha

        q, alpha = scan(loop1, q, states, reverse=True)

        gamma = 1 if k == 0 else (s[k - 1].T @ y[k - 1]) / (y[k - 1].T @ y[k - 1])
        z = gamma * q
        states = jnp.concatenate((states, alpha[:, None]), axis=1)

        def loop2(z, state):
            s = state[:n]
            y = state[n : 2 * n]
            rho = state[2 * n]
            alpha = state[2 * n + 1]
            beta = rho * y @ z
            z += s * (alpha - beta)
            return z, None

        z, _ = scan(loop2, z, states)

        w_new = w - lr * z
        g_new = grad_f(w_new, *args)

        if len(s) == m:
            s = s[1:]
            y = y[1:]
            rho = rho[1:]

        s = jnp.concatenate((s, (w_new - w)[None]))
        y = jnp.concatenate((y, (g_new - g)[None]))
        rho = jnp.concatenate((rho, (1 / (y[-1] @ s[-1])).reshape(1, 1)))

        return w_new, g_new, s, y, rho

    w = w0
    g0 = grad_f(w0, *args)
    w = w0 - g0
    g = grad_f(w, *args)

    s: Array = (w - w0)[None]
    y: Array = (g - g0)[None]
    rho: Array = (1 / (y[-1] @ s[-1])).reshape(1, 1)

    for k in range(n_iters - 1):
        w, g, s, y, rho = lbfgs_step(w, g, s, y, rho)

    return w
