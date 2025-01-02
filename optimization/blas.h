#include <Accelerate/Accelerate.h>

// axpy template
template <typename T>
void axpy(const int n, const T alpha, const T* x, const int incx, T* y, const int incy);

template <>
inline void axpy<float>(const int n, const float alpha, const float* x, const int incx, float* y, const int incy) {
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void axpy<double>(const int n, const double alpha, const double* x, const int incx, double* y, const int incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

// dot template
template <typename T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy);

template <>
inline float dot<float>(const int n, const float* x, const int incx, const float* y, const int incy) {
    return cblas_sdot(n, x, incx, y, incy);
}

template <>
inline double dot<double>(const int n, const double* x, const int incx, const double* y, const int incy) {
    return cblas_ddot(n, x, incx, y, incy);
}

// scal template
template <typename T>
void scal(const int n, const T alpha, T* x, const int incx);

template <>
inline void scal<float>(const int n, const float alpha, float* x, const int incx) {
    cblas_sscal(n, alpha, x, incx);
}

template <>
inline void scal<double>(const int n, const double alpha, double* x, const int incx) {
    cblas_dscal(n, alpha, x, incx);
}

// nrm2 template
template <typename T>
T nrm2(const int n, const T* x, const int incx);

template <>
inline float nrm2<float>(const int n, const float* x, const int incx) {
    return cblas_snrm2(n, x, incx);
}

template <>
inline double nrm2<double>(const int n, const double* x, const int incx) {
    return cblas_dnrm2(n, x, incx);
}