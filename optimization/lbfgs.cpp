#include <tuple>
#include <algorithm>
#include <string.h>
#include <Accelerate/Accelerate.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>


namespace py = pybind11;


std::tuple<float, float*> value_and_grad_float32(
    py::function py_value_and_grad, 
    const float* x,
    const int n
) {
    auto result = py_value_and_grad(py::array_t<float>(n, x));
    if (!py::isinstance<py::tuple>(result)) {
        throw std::invalid_argument("Python function must return a tuple.");
    }

    auto result_tuple = result.cast<py::tuple>();
    if (result_tuple.size() != 2) {
        throw std::invalid_argument("Tuple must have exactly two elements.");
    }

    py::array_t<float> value_array = result_tuple[0].cast<py::array_t<float>>();
    py::array_t<float> grad_array = result_tuple[1].cast<py::array_t<float>>();

    if (value_array.ndim() != 0 || grad_array.ndim() != 1) {
        throw std::invalid_argument("Returned arrays must be 1-dimensional.");
    }

    float value = *value_array.mutable_data();
    auto grad_info = grad_array.request();
    float* grad = new float[n];
    memcpy(grad, static_cast<float*>(grad_info.ptr), n * sizeof(float));
    return {value, grad};
}

std::tuple<float, float, float*> backtracking_line_search_float32(
    py::function py_value_and_grad, 
    const float* x0,
    const float f0,
    const float* g0,
    const float* d,
    float alpha,
    float beta,
    const int max_iters,
    const int n
) {
    float t = 1;
    float* x = new float[n];
    memcpy(x, x0, n * sizeof(float));
    cblas_saxpy(n, t, d, 1, x, 1);
    auto [f, g] = value_and_grad_float32(py_value_and_grad, x, n);
    for (int i = 0; i < max_iters; i++) {
        if (f <= f0 + alpha * t * cblas_sdot(n, g0, 1, d, 1)) {
            break;
        }
        t *= beta;
        memcpy(x, x0, n * sizeof(float));
        cblas_saxpy(n, t, d, 1, x, 1);
        std::tie(f, g) = value_and_grad_float32(py_value_and_grad, x, n);
    }
    delete[] x;
    return {t, f, g};
}

float* lbfgs_hvp_approx_float32(
    float* q,
    const float* s,
    const float* y,
    const float* rho,
    float* alpha,
    const int k,
    const int history_size,
    const int n
) {
    for (int i = k - 1; i >= std::max(k - history_size, 0); i--) {
        int j = i % history_size;
        alpha[j] = rho[j] * cblas_sdot(n, s + (j * n), 1, q, 1);
        cblas_saxpy(n, -alpha[j], y + (j * n), 1, q, 1);
    }
    float* z = q;
    if (k > 0) {
        int j = (k - 1) % history_size;
        float a = cblas_sdot(n, s + (j * n), 1, y + (j * n), 1);
        float b = cblas_sdot(n, y + (j * n), 1, y + (j * n), 1);
        cblas_sscal(n, a / b, z, 1);
    }
    for (int i = std::max(k - history_size, 0); i < k; i++) {
        int j = i % history_size;
        float beta = rho[j] * cblas_sdot(n, y + (j * n), 1, z, 1);
        cblas_saxpy(n, alpha[j] - beta, s + (j * n), 1, z, 1);
    }
    return z;
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
    if (x0.ndim() != 1) {
        throw std::invalid_argument("x must be 1-dimensional.");
    }
    const int n = x0.size();
    auto x0_info = x0.request();
    float* x = new float[n];
    memcpy(x, x0_info.ptr, n * sizeof(float));

    float* s = new float[history_size * n];
    float* y = new float[history_size * n];
    float* rho = new float[history_size];
    float* hvp_alpha = new float[history_size];
    float* q = new float[n];

    auto [f, g] = value_and_grad_float32(py_value_and_grad, x, n);

    for (int k = 0; k < max_iters; k++) {
        int j = k % history_size;
        memcpy(q, g, n * sizeof(float));
        float* z = lbfgs_hvp_approx_float32(q, s, y, rho, hvp_alpha, k, history_size, n);

        cblas_sscal(n, -1, z, 1);
        auto [t, f_new, g_new] = backtracking_line_search_float32(
            py_value_and_grad, x, f, g, z, alpha, beta, max_ls_iters, n
        );
        cblas_sscal(n, t, z, 1);
        cblas_saxpy(n, 1, z, 1, x, 1);
        
        if (cblas_snrm2(n, g_new, 1) < tol) {
            break;
        }

        if (k < max_iters - 1) {
            memcpy(s + (j * n), z, n * sizeof(float));
            memcpy(y + (j * n), g_new, n * sizeof(float));
            cblas_saxpy(n, -1, g, 1, y + (j * n), 1);
            rho[j] = 1 / (cblas_sdot(n, s + (j * n), 1, y + (j * n), 1));
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

    return py::array_t<float>(n, x);
}

std::tuple<double, double*> value_and_grad_float64(
    py::function py_value_and_grad, 
    const double* x,
    const int n
) {
    auto result = py_value_and_grad(py::array_t<double>(n, x));
    if (!py::isinstance<py::tuple>(result)) {
        throw std::invalid_argument("Python function must return a tuple.");
    }

    auto result_tuple = result.cast<py::tuple>();
    if (result_tuple.size() != 2) {
        throw std::invalid_argument("Tuple must have exactly two elements.");
    }

    py::array_t<double> value_array = result_tuple[0].cast<py::array_t<double>>();
    py::array_t<double> grad_array = result_tuple[1].cast<py::array_t<double>>();

    if (value_array.ndim() != 0 || grad_array.ndim() != 1) {
        throw std::invalid_argument("Returned arrays must be 1-dimensional.");
    }

    double value = *value_array.mutable_data();
    auto grad_info = grad_array.request();
    double* grad = new double[n];
    memcpy(grad, static_cast<double*>(grad_info.ptr), n * sizeof(double));
    return {value, grad};
}

std::tuple<double, double, double*> backtracking_line_search_float64(
    py::function py_value_and_grad, 
    const double* x0,
    const double f0,
    const double* g0,
    const double* d,
    double alpha,
    double beta,
    const int max_iters,
    const int n
) {
    double t = 1;
    double* x = new double[n];
    memcpy(x, x0, n * sizeof(double));
    cblas_daxpy(n, t, d, 1, x, 1);
    auto [f, g] = value_and_grad_float64(py_value_and_grad, x, n);
    for (int i = 0; i < max_iters; i++) {
        if (f <= f0 + alpha * t * cblas_ddot(n, g0, 1, d, 1)) {
            break;
        }
        t *= beta;
        memcpy(x, x0, n * sizeof(double));
        cblas_daxpy(n, t, d, 1, x, 1);
        std::tie(f, g) = value_and_grad_float64(py_value_and_grad, x, n);
    }
    delete[] x;
    return {t, f, g};
}

double* lbfgs_hvp_approx_float64(
    double* q,
    const double* s,
    const double* y,
    const double* rho,
    double* alpha,
    const int k,
    const int history_size,
    const int n
) {
    for (int i = k - 1; i >= std::max(k - history_size, 0); i--) {
        int j = i % history_size;
        alpha[j] = rho[j] * cblas_ddot(n, s + (j * n), 1, q, 1);
        cblas_daxpy(n, -alpha[j], y + (j * n), 1, q, 1);
    }
    double* z = q;
    if (k > 0) {
        int j = (k - 1) % history_size;
        double a = cblas_ddot(n, s + (j * n), 1, y + (j * n), 1);
        double b = cblas_ddot(n, y + (j * n), 1, y + (j * n), 1);
        cblas_dscal(n, a / b, z, 1);
    }
    for (int i = std::max(k - history_size, 0); i < k; i++) {
        int j = i % history_size;
        double beta = rho[j] * cblas_ddot(n, y + (j * n), 1, z, 1);
        cblas_daxpy(n, alpha[j] - beta, s + (j * n), 1, z, 1);
    }
    return z;
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
    if (x0.ndim() != 1) {
        throw std::invalid_argument("x must be 1-dimensional.");
    }
    const int n = x0.size();
    auto x0_info = x0.request();
    double* x = new double[n];
    memcpy(x, x0_info.ptr, n * sizeof(double));

    double* s = new double[history_size * n];
    double* y = new double[history_size * n];
    double* rho = new double[history_size];
    double* hvp_alpha = new double[history_size];
    double* q = new double[n];

    auto [f, g] = value_and_grad_float64(py_value_and_grad, x, n);

    for (int k = 0; k < max_iters; k++) {
        int j = k % history_size;
        memcpy(q, g, n * sizeof(double));
        double* z = lbfgs_hvp_approx_float64(q, s, y, rho, hvp_alpha, k, history_size, n);

        cblas_dscal(n, -1, z, 1);
        auto [t, f_new, g_new] = backtracking_line_search_float64(
            py_value_and_grad, x, f, g, z, alpha, beta, max_ls_iters, n
        );
        cblas_dscal(n, t, z, 1);
        cblas_daxpy(n, 1, z, 1, x, 1);
        
        if (cblas_dnrm2(n, g_new, 1) < tol) {
            break;
        }

        if (k < max_iters - 1) {
            memcpy(s + (j * n), z, n * sizeof(double));
            memcpy(y + (j * n), g_new, n * sizeof(double));
            cblas_daxpy(n, -1, g, 1, y + (j * n), 1);
            rho[j] = 1 / (cblas_ddot(n, s + (j * n), 1, y + (j * n), 1));
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

    return py::array_t<double>(n, x);
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
