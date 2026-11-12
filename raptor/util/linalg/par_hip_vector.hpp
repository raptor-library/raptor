// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_UTIL_PAR_HIP_VECTOR_HPP
#define RAPTOR_UTIL_PAR_HIP_VECTOR_HPP

#include "hip_vector.hpp"

namespace raptor
{

// global 2-norm
double par_hip_norm(const double* x, int n, rocblas_handle handle);

// global dot product
double par_hip_inner_product(const double* x, const double* y, int n,
        rocblas_handle handle);
} // namespace raptor
#endif
