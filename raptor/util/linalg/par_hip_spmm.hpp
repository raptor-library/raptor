// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_UTIL_PAR_HIP_SPMM_HPP
#define RAPTOR_UTIL_PAR_HIP_SPMM_HPP

#include "raptor/core/par_hip_matrix.hpp"
#include "raptor/core/par_hip_multivector.hpp"
#include <rocblas/rocblas.h>

namespace raptor
{
// Parallel B = alpha*A*X + beta*B in column-major storage
void par_hip_spmm(double alpha, ParHipCSRMatrix& A, const ParHipMultiVectorView& X,
        double beta, const ParHipMultiVectorView& B, rocsparse_handle sparse_handle,
        rocblas_handle blas_handle);

// Parallel R = B - A*X for nrhs packed column-major vectors
void par_hip_spmm_residual(ParHipCSRMatrix& A, const ParHipMultiVectorView& X,
        const ParHipMultiVectorView& B, const ParHipMultiVectorView& R,
        rocsparse_handle sparse_handle, rocblas_handle blas_handle);
} // namespace raptor
#endif
