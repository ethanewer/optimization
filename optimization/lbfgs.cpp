#include <tuple>
#include <algorithm>
#include <string.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <blas.h>

namespace py = pybind11;

template <typename T>
std::tuple<T, T*> value_and_grad(
    py::function py_value_and_grad, 
    const T* x,
    const int n
) {
    auto result = py_value_and_grad(py::array_t<T>(n, x));
    if (!py::isinstance<py::tuple>(result)) {
        throw std::invalid_argument("Python function must return a tuple.");
    }

    auto result_tuple = result.template cast<py::tuple>();
    if (result_tuple.size() != 2) {
        throw std::invalid_argument("Tuple must have exactly two elements.");
    }

    py::array_t<T> value_array = result_tuple[0].template cast<py::array_t<T>>();
    py::array_t<T> grad_array = result_tuple[1].template cast<py::array_t<T>>();

    if (value_array.ndim() != 0 || grad_array.ndim() != 1) {
        throw std::invalid_argument("Returned arrays must be 1-dimensional.");
    }

    T value = *value_array.mutable_data();
    auto grad_info = grad_array.request();
    T* grad = new T[n];
    memcpy(grad, static_cast<T*>(grad_info.ptr), n * sizeof(T));
    return {value, grad};
}

template <typename T>
std::tuple<T, T, T*> backtracking_line_search(
    py::function py_value_and_grad, 
    const T* x0,
    const T f0,
    const T* g0,
    const T* d,
    T alpha,
    T beta,
    const int max_iters,
    const int n
) {
    T t = 1;
    T* x = new T[n];
    memcpy(x, x0, n * sizeof(T));
    axpy(n, t, d, 1, x, 1);
    auto [f, g] = value_and_grad(py_value_and_grad, x, n);
    for (int i = 0; i < max_iters; i++) {
        if (f <= f0 + alpha * t * dot(n, g0, 1, d, 1)) {
            break;
        }
        t *= beta;
        memcpy(x, x0, n * sizeof(T));
        axpy(n, t, d, 1, x, 1);
        std::tie(f, g) = value_and_grad(py_value_and_grad, x, n);
    }
    delete[] x;
    return {t, f, g};
}

template <typename T>
T* lbfgs_hvp_approx(
    T* q,
    const T* s,
    const T* y,
    const T* rho,
    T* alpha,
    const int k,
    const int history_size,
    const int n
) {
    for (int i = k - 1; i >= std::max(k - history_size, 0); i--) {
        int j = i % history_size;
        alpha[j] = rho[j] * dot(n, s + (j * n), 1, q, 1);
        axpy(n, -alpha[j], y + (j * n), 1, q, 1);
    }
    T* z = q;
    if (k > 0) {
        int j = (k - 1) % history_size;
        T a = dot(n, s + (j * n), 1, y + (j * n), 1);
        T b = dot(n, y + (j * n), 1, y + (j * n), 1);
        scal(n, a / b, z, 1);
    }
    for (int i = std::max(k - history_size, 0); i < k; i++) {
        int j = i % history_size;
        T beta = rho[j] * dot(n, y + (j * n), 1, z, 1);
        axpy(n, alpha[j] - beta, s + (j * n), 1, z, 1);
    }
    return z;
}

template <typename T>
py::array_t<T> fmin_lbfgs(
    py::function py_value_and_grad, 
    py::array_t<T> x0,
    const int history_size,
    T tol,
    const int max_iters,
    T alpha,
    T beta,
    const int max_ls_iters
) { 
    if (x0.ndim() != 1) {
        throw std::invalid_argument("x must be 1-dimensional.");
    }
    const int n = x0.size();
    auto x0_info = x0.request();
    T* x = new T[n];
    memcpy(x, x0_info.ptr, n * sizeof(T));

    T* s = new T[history_size * n];
    T* y = new T[history_size * n];
    T* rho = new T[history_size];
    T* hvp_alpha = new T[history_size];
    T* q = new T[n];

    auto [f, g] = value_and_grad(py_value_and_grad, x, n);

    for (int k = 0; k < max_iters; k++) {
        int j = k % history_size;
        memcpy(q, g, n * sizeof(T));
        T* z = lbfgs_hvp_approx(q, s, y, rho, hvp_alpha, k, history_size, n);

        scal(n, static_cast<T>(-1), z, 1);
        auto [t, f_new, g_new] = backtracking_line_search(
            py_value_and_grad, x, f, g, z, alpha, beta, max_ls_iters, n
        );
        scal(n, t, z, 1);
        axpy(n, static_cast<T>(1), z, 1, x, 1);
        
        if (nrm2(n, g_new, 1) < tol) {
            break;
        }

        if (k < max_iters - 1) {
            memcpy(s + (j * n), z, n * sizeof(T));
            memcpy(y + (j * n), g_new, n * sizeof(T));
            axpy(n, static_cast<T>(-1), g, 1, y + (j * n), 1);
            rho[j] = 1 / (dot(n, s + (j * n), 1, y + (j * n), 1));
            f = f_new;
            delete[] g;
            g = g_new;
        }
    }

    delete[] s;
    delete[] y;
    delete[] rho;
    delete[] hvp_alpha;
    delete[] q;
    delete[] g;

    return py::array_t<T>(n, x);
}

py::array_t<float> fmin_lbfgs_float32(
    py::function py_value_and_grad, 
    py::array_t<float> x0,
    const int history_size,
    float tol,
    const int max_iters,
    float alpha,
    float beta,
    const int max_ls_iters
) { 
    return fmin_lbfgs(
        py_value_and_grad,
        x0,
        history_size,
        tol,
        max_iters,
        alpha,
        beta,
        max_ls_iters
    );
}

py::array_t<double> fmin_lbfgs_float64(
    py::function py_value_and_grad, 
    py::array_t<double> x0,
    const int history_size,
    double tol,
    const int max_iters,
    double alpha,
    double beta,
    const int max_ls_iters
) { 
    return fmin_lbfgs(
        py_value_and_grad,
        x0,
        history_size,
        tol,
        max_iters,
        alpha,
        beta,
        max_ls_iters
    );
}

PYBIND11_MODULE(lbfgs, m) {
    m.def("fmin_lbfgs_float32", &fmin_lbfgs_float32,
        "Minimize a function using LBFGS.",
        py::arg("value_and_grad"), 
        py::arg("x0"),
        py::arg("history_size"),
        py::arg("tol"),
        py::arg("max_iters"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("max_ls_iters")
    );
    m.def("fmin_lbfgs_float64", &fmin_lbfgs_float64,
        "Minimize a function using LBFGS.",
        py::arg("value_and_grad"), 
        py::arg("x0"),
        py::arg("history_size"),
        py::arg("tol"),
        py::arg("max_iters"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("max_ls_iters")
    );
}
