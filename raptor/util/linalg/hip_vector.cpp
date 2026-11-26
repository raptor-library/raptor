// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "hip_vector.hpp"
#include "raptor/core/hip_error.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace raptor
{
void hip_copy(double* dst, const double* src, std::size_t n, hipStream_t stream)
{
    if (n == 0 || dst == src) return;

    HIP_CHECK(hipMemcpyAsync(dst, src, n * sizeof(double),
            hipMemcpyDeviceToDevice, stream));
}

void hip_copy(const HipMultiVectorView& dst, const HipMultiVectorView& src, hipStream_t stream)
{
    if (dst.rows < 0 || dst.cols < 0 || dst.rows != src.rows || dst.cols != src.cols)
        throw std::invalid_argument("HIP multivector copy requires matching dimensions");

    hip_copy(dst.data, src.data, static_cast<std::size_t>(dst.rows) * dst.cols, stream);
}

void hip_mult(double alpha, const HipMultiVectorView& P, const HipMultiVectorView& Q, double beta,
              HipMultiVectorView Y, rocblas_handle handle)
{
    if (P.rows < 0 || P.cols < 0 || Q.cols < 0 ||
        Q.rows != P.cols || Y.rows != P.rows || Y.cols != Q.cols)
        throw std::invalid_argument("Incompatible HIP multivector shapes");

    if (Y.rows == 0 || Y.cols == 0) return;

    if (Y.data && (Y.data == P.data || Y.data == Q.data))
        throw std::invalid_argument("HIP GEMM output must use separate storage");

    // Stride needs to be > 0 
    ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            P.rows, Q.cols, P.cols,
            &alpha, P.data, P.rows,
            Q.data, std::max(1, Q.rows),
            &beta, Y.data, Y.rows));
}

void hip_mult_T(double alpha, const HipMultiVectorView& P, const HipMultiVectorView& Q,
        double beta, HipMultiVectorView Y, rocblas_handle handle)
{
    if (P.rows < 0 || P.cols < 0 || Q.cols < 0 ||
        Q.rows != P.rows || Y.rows != P.cols || Y.cols != Q.cols)
        throw std::invalid_argument("Incompatible HIP multivector shapes");

    if (Y.rows == 0 || Y.cols == 0) return;

    if (Y.data && (Y.data == P.data || Y.data == Q.data))
        throw std::invalid_argument("HIP GEMM output must use separate storage");

    // Stride needs to be > 0 
    ROCBLAS_CHECK(rocblas_dgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            P.cols, Q.cols, P.rows,
            &alpha, P.data, std::max(1, Q.rows), 
            Q.data, std::max(1, Q.rows),
            &beta, Y.data, Y.rows));
}

double hip_norm(const double* x, int n, rocblas_handle handle)
{
    return std::sqrt(hip_inner_product(x, x, n, handle));
}

double hip_inner_product(const double* x, const double* y, int n,
        rocblas_handle handle)
{
    double inner_prod;
    ROCBLAS_CHECK(rocblas_ddot(handle, n, x, 1, y, 1, &inner_prod));
    HIP_CHECK(hipDeviceSynchronize());
    return inner_prod;
}

void hip_axpy(double alpha, double* x, const double* y, int n,
        rocblas_handle handle)
{
    ROCBLAS_CHECK(rocblas_daxpy(handle, n, &alpha, y, 1, x, 1));
}

void hip_scale(double alpha, double* x, int n, rocblas_handle handle)
{
    ROCBLAS_CHECK(rocblas_dscal(handle, n, &alpha, x, 1));
}
} // namespace raptor
