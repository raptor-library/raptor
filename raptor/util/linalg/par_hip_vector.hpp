// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_UTIL_PAR_HIP_VECTOR_HPP
#define RAPTOR_UTIL_PAR_HIP_VECTOR_HPP

#include "hip_vector.hpp"
#include "raptor/core/par_hip_multivector.hpp"

namespace raptor
{

// parallel Y = alpha*P*Q + beta*Y for packed column-major multivectors P, Q, Y; Q is owned locally
void par_hip_mult(double alpha, const ParHipMultiVectorView& P,
        const HipMultiVectorView& Q, double beta, ParHipMultiVectorView Y,
        rocblas_handle handle);

// parallel Y = alpha*P^T*Q + beta*Y ocolumn-major multivectors P, Q, Y; Y is owned locally
void par_hip_mult_T(double alpha, const ParHipMultiVectorView& P,
        const ParHipMultiVectorView& Q, double beta, HipMultiVectorView Y,
        double* host_buffer, rocblas_handle handle);

// global 2-norm
double par_hip_norm(const double* x, int n, rocblas_handle handle);

// global dot product
double par_hip_inner_product(const double* x, const double* y, int n,
        rocblas_handle handle);
} // namespace raptor
#endif
