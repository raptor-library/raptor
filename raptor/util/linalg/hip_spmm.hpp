// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_UTIL_HIP_SPMM_HPP
#define RAPTOR_UTIL_HIP_SPMM_HPP

#include "raptor/core/hip_matrix.hpp"
#include <rocblas/rocblas.h>

namespace raptor
{
// B = alpha*A*X + beta*B for nrhs packed column-major vectors
void hip_spmm(double alpha, HipCSRMatrix& A, const double* X,
        double beta, double* B, int nrhs, rocsparse_handle sparse_handle,
        rocblas_handle blas_handle);

// R = B - A*X for nrhs packed column-major vectors
void hip_spmm_residual(HipCSRMatrix& A, const double* X, const double* B,
        double* R, int nrhs, rocsparse_handle sparse_handle, rocblas_handle blas_handle);
} // namespace raptor
#endif
