// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "hip_vector.hpp"
#include "raptor/core/hip_error.hpp"
#include <cmath>

namespace raptor
{
void hip_copy(double* dst, const double* src, std::size_t n, hipStream_t stream)
{
    if (n == 0 || dst == src) return;

    HIP_CHECK(hipMemcpyAsync(dst, src, n * sizeof(double),
            hipMemcpyDeviceToDevice, stream));
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
