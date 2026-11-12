// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_KRYLOV_PAR_HIP_CG_HPP
#define RAPTOR_KRYLOV_PAR_HIP_CG_HPP

#include <vector>
#include <rocblas/rocblas.h>
#include "raptor/core/par_hip_matrix.hpp"

namespace raptor
{

void CG(ParHipCSRMatrix* A, double* x, const double* b, std::vector<double>& res,
        rocsparse_handle sparse_handle, rocblas_handle blas_handle,
        double tol = 1e-05, int max_iter = -1, double* comm_t = nullptr);
} // namespace raptor
#endif
