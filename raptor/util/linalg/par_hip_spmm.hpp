// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_UTIL_PAR_HIP_SPMM_HPP
#define RAPTOR_UTIL_PAR_HIP_SPMM_HPP

#include "raptor/core/par_hip_matrix.hpp"
#include <rocblas/rocblas.h>

namespace raptor
{
// Parallel B = alpha*A*X + beta*B in column-major storage
void par_hip_spmm(double alpha, ParHipCSRMatrix& A, const double* X,
        double beta, double* B, int nrhs, rocsparse_handle sparse_handle,
        rocblas_handle blas_handle);

// Parallel R = B - A*X for nrhs packed column-major vectors
void par_hip_spmm_residual(ParHipCSRMatrix& A, const double* X, const double* B,
        double* R, int nrhs, rocsparse_handle sparse_handle, rocblas_handle blas_handle);
} // namespace raptor
#endif
