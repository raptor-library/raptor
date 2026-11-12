// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "par_hip_vector.hpp"
#include "raptor/core/mpi_types.hpp"
#include <cmath>

namespace raptor
{
double par_hip_norm(const double* x, int n, rocblas_handle handle)
{
    return std::sqrt(par_hip_inner_product(x, x, n, handle));
}

double par_hip_inner_product(const double* x, const double* y, int n,
        rocblas_handle handle)
{
    double inner_prod = hip_inner_product(x, y, n, handle);
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, &inner_prod, 1,
            RAPtor_MPI_DOUBLE, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    return inner_prod;
}
} // namespace raptor
