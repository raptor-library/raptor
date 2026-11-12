// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_UTIL_PAR_HIP_SPMV_HPP
#define RAPTOR_UTIL_PAR_HIP_SPMV_HPP

#include "hip_spmv.hpp"
#include "raptor/core/par_hip_matrix.hpp"

namespace raptor
{
// Parallel b = alpha*A*x + beta*b 
void par_hip_spmv(double alpha, ParHipCSRMatrix& A, const double* x,
        double beta, double* b, rocsparse_handle handle);

// Parallel r = b - A*x 
void par_hip_residual(ParHipCSRMatrix& A, const double* x, const double* b,
        double* r, rocsparse_handle handle);
} // namespace raptor
#endif
