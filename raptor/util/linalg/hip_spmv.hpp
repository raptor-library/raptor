// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_UTIL_HIP_SPMV_HPP
#define RAPTOR_UTIL_HIP_SPMV_HPP

#include "raptor/core/hip_matrix.hpp"

namespace raptor
{
// b = alpha*A*x + beta*b 
void hip_spmv(double alpha, HipCSRMatrix& A, const double* x,
        double beta, double* b, rocsparse_handle handle);

// r = b - A*x 
void hip_residual(HipCSRMatrix& A, const double* x, const double* b,
        double* r, rocsparse_handle handle);
} // namespace raptor
#endif
