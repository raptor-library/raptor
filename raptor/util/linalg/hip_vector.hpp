// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_UTIL_HIP_VECTOR_HPP
#define RAPTOR_UTIL_HIP_VECTOR_HPP

#include "raptor/core/hip_multivector.hpp"
#include <cstddef>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace raptor
{

// Enqueue a device-to-device copy on stream
void hip_copy(double* dst, const double* src, std::size_t n, hipStream_t stream);

// Copy packed multivectors with matching dimensions
void hip_copy(const HipMultiVectorView& dst, const HipMultiVectorView& src, hipStream_t stream);

// Y = alpha*P*Q + beta*Y for packed column-major multivectors P, Q, Y
void hip_mult(double alpha, const HipMultiVectorView& P,const HipMultiVectorView& Q, double beta,
              HipMultiVectorView Y, rocblas_handle handle);

// Y = alpha*P^T*Q + beta*Y for packed column-major multivectors P, Q, Y
void hip_mult_T(double alpha, const HipMultiVectorView& P, const HipMultiVectorView& Q,
        double beta, HipMultiVectorView Y, rocblas_handle handle);

// Local 2-norm
double hip_norm(const double* x, int n, rocblas_handle handle);

// Local dot product
double hip_inner_product(const double* x, const double* y, int n,
        rocblas_handle handle);

// x += alpha*y
void hip_axpy(double alpha, double* x, const double* y, int n,
        rocblas_handle handle);

// x *= alpha
void hip_scale(double alpha, double* x, int n, rocblas_handle handle);
} // namespace raptor
#endif
