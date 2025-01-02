from typing import Any, Callable, Sequence

import numpy as np
from lbfgs import fmin_lbfgs_float32, fmin_lbfgs_float64


def fmin_lbfgs(
    value_and_grad: Callable[..., tuple[np.ndarray, np.ndarray]],
    x: np.ndarray,
    args: Sequence[Any] = (),
    history_size: int = 20,
    tol: float = 1e-4,
    max_iters: int = 50,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_ls_iters: int = 10,
) -> np.ndarray:  # type: ignore
    if x.dtype == np.float32:
        return fmin_lbfgs_float32(
            lambda x: value_and_grad(x, *args),
            x,
            history_size,
            tol,
            max_iters,
            alpha,
            beta,
            max_ls_iters,
        )
    elif x.dtype == np.float64:
        return fmin_lbfgs_float64(
            lambda x: value_and_grad(x, *args),
            x,
            history_size,
            tol,
            max_iters,
            alpha,
            beta,
            max_ls_iters,
        )
    else:
        raise NotImplementedError("x must be a float32 or float64 array.")


# def backtracking_line_search(
#     value_and_grad: Callable[..., tuple[float, np.ndarray]],
#     x0: np.ndarray,
#     f0: float,
#     g0: np.ndarray,
#     d: np.ndarray,
#     alpha: float,
#     beta: float,
#     max_iters: int,
#     args: Sequence[Any] = (),
# ) -> tuple[float, float, np.ndarray]:
#     t = 1.0
#     f, g = value_and_grad(x0 + t * d, *args)
#     for _ in range(max_iters):
#         if f <= f0 + alpha * t * g0 @ d:
#             break

#         t *= beta
#         f, g = value_and_grad(x0 + t * d, *args)

#     return t, f, g


# def lbfgs_hvp_approx(
#     q: np.ndarray,
#     s: np.ndarray,
#     y: np.ndarray,
#     rho: np.ndarray,
#     k: int,
#     history_size: int,
# ) -> np.ndarray:
#     alpha = np.ones(history_size, dtype=q.dtype)
#     for i in range(k - 1, max(k - history_size - 1, -1), -1):
#         j = i % history_size
#         alpha[j] = rho[j] * s[j] @ q
#         q -= alpha[j] * y[j]

#     j = (k - 1) % history_size
#     gamma = (s[j] @ y[j]) / (y[j] @ y[j])
#     z = gamma * q

#     for i in range(max(k - history_size, 0), k):
#         j = i % history_size
#         beta = rho[j] * y[j] @ z
#         z += s[j] * (alpha[j] - beta)

#     return z


# def fmin_lbfgs(
#     value_and_grad: Callable[..., tuple[float, np.ndarray]],
#     x: np.ndarray,
#     args: Sequence[Any] = (),
#     history_size: int = 20,
#     tol: float = 1e-4,
#     max_iters: int = 50,
#     alpha: float = 0.1,
#     beta: float = 0.5,
#     max_ls_iters: int = 10,
# ) -> np.ndarray:
#     if x.dtype == np.float32:
#         return fmin_lbfgs_float32(
#             lambda x: value_and_grad(x, *args),
#             x,
#             history_size,
#             tol,
#             max_iters,
#             alpha,
#             beta,
#             max_ls_iters,
#         )
#     elif x.dtype == np.float64:
#         fmin_lbfgs_float64(
#             lambda x: value_and_grad(x, *args),
#             x,
#             history_size,
#             tol,
#             max_iters,
#             alpha,
#             beta,
#             max_ls_iters,
#         )

#     s = np.ones((history_size, len(x)), dtype=x.dtype)
#     y = np.ones((history_size, len(x)), dtype=x.dtype)
#     rho = np.ones(history_size, dtype=x.dtype)

#     f, g = value_and_grad(x, *args)

#     for k in range(max_iters):
#         j = k % history_size
#         z = lbfgs_hvp_approx(g.copy(), s, y, rho, k, history_size)
#         t, f, g_new = backtracking_line_search(
#             value_and_grad, x, f, g, -z, alpha, beta, max_ls_iters, args
#         )
#         x -= t * z

#         if np.linalg.norm(g) < tol:
#             break

#         if k < max_iters - 1:
#             s[j] = -t * z
#             y[j] = g_new - g
#             rho[j] = 1 / (y[j] @ s[j])
#             g = g_new

#     return x
