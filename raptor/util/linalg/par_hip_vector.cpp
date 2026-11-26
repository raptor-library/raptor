// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "par_hip_vector.hpp"
#include "raptor/core/hip_error.hpp"
#include "raptor/core/mpi_types.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace raptor
{
void par_hip_mult(double alpha, const ParHipMultiVectorView& P,
        const HipMultiVectorView& Q, double beta, ParHipMultiVectorView Y,
        rocblas_handle handle)
{
    if (P.global_rows != Y.global_rows)
        throw std::invalid_argument("Parallel HIP multivector multiply requires compatible global row counts");

    hip_mult(alpha, P.local, Q, beta, Y.local, handle);
}

void par_hip_mult_T(double alpha, const ParHipMultiVectorView& P,
        const ParHipMultiVectorView& Q, double beta, HipMultiVectorView Y,
        double* host_buffer, rocblas_handle handle)
{
    if (P.global_rows != Q.global_rows || P.local.rows != Q.local.rows ||
        Y.rows < 0 || Y.cols < 0 || Y.rows != P.local.cols || Y.cols != Q.local.cols)
    {
        std::fprintf(stderr, "Incompatible parallel HIP transpose multiply shapes\n");
        RAPTOR_HIP_ABORT();
        return;
    }
    if (Y.rows == 0 || Y.cols == 0) return;

    // TODO: we can probably simplify these guards to incorporate into the multivector struct
    if (!host_buffer || !Y.data || Y.data == P.local.data || Y.data == Q.local.data ||
        Y.rows > std::numeric_limits<int>::max() / Y.cols)
    {
        std::fprintf(stderr, "Invalid buffers or MPI count for parallel HIP transpose multiply\n");
        RAPTOR_HIP_ABORT();
        return;
    }

    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    // Only rank 0 contributes the existing Y to the sum
    hip_mult_T(alpha, P.local, Q.local, rank == 0 ? beta : 0.0, Y, handle);

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));
    const int count = Y.rows * Y.cols;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(double);
    HIP_CHECK(hipMemcpyWithStream(host_buffer, Y.data, bytes, hipMemcpyDeviceToHost, stream));
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, host_buffer, count,
            RAPtor_MPI_DOUBLE, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    HIP_CHECK(hipMemcpyWithStream(Y.data, host_buffer, bytes, hipMemcpyHostToDevice, stream));
}

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
