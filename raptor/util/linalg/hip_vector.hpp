// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_UTIL_HIP_VECTOR_HPP
#define RAPTOR_UTIL_HIP_VECTOR_HPP

#include <rocblas/rocblas.h>

namespace raptor
{

// Copy y to x
void hip_copy(double* x, const double* y, int n);

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
