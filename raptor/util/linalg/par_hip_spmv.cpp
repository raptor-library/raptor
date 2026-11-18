// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "par_hip_spmv.hpp"
#include "hip_vector.hpp"
#include "raptor/core/hip_error.hpp"
#include "raptor/core/comm_pkg.hpp"
#include <vector>

namespace raptor
{
void par_hip_residual(ParHipCSRMatrix& A, const double* x, const double* b,
        double* r, rocsparse_handle sparse_handle, rocblas_handle blas_handle)
{
    hip_copy(r, b, A.on_proc.n_rows);
    par_hip_spmv(-1.0, A, x, 1.0, r, sparse_handle, blas_handle);
}

void par_hip_spmv(double alpha, ParHipCSRMatrix& A, const double* x,
        double beta, double* b, rocsparse_handle sparse_handle, rocblas_handle blas_handle)
{
    auto& send = *A.comm->send_data;
    auto& recv = *A.comm->recv_data;

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));
    if (send.size_msgs > 0)
    {
        std::vector<double> h_x(A.on_proc.n_cols);
        HIP_CHECK(hipMemcpyWithStream(h_x.data(), x, h_x.size() * sizeof(double),
                hipMemcpyDeviceToHost, stream));
        for (int i = 0; i < send.size_msgs; ++i)
            A.h_sendbuf[i] = h_x[send.indices[i]];
    }

    for (int i = 0; i < recv.num_msgs; ++i)
        RAPtor_MPI_Irecv(A.h_recvbuf + recv.indptr[i],
                recv.indptr[i + 1] - recv.indptr[i], RAPtor_MPI_DOUBLE,
                recv.procs[i], A.comm->key, A.comm->mpi_comm, &recv.requests[i]);
    for (int i = 0; i < send.num_msgs; ++i)
        RAPtor_MPI_Isend(A.h_sendbuf + send.indptr[i],
                send.indptr[i + 1] - send.indptr[i], RAPtor_MPI_DOUBLE,
                send.procs[i], A.comm->key, A.comm->mpi_comm, &send.requests[i]);

    hip_spmv(alpha, A.on_proc, x, beta, b, sparse_handle, blas_handle);
    if (recv.num_msgs)
        RAPtor_MPI_Waitall(recv.num_msgs, recv.requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    if (recv.size_msgs > 0)
        HIP_CHECK(hipMemcpyWithStream(A.d_recvbuf, A.h_recvbuf,
                static_cast<std::size_t>(recv.size_msgs) * sizeof(double),
                hipMemcpyHostToDevice, stream));
    hip_spmv(alpha, A.off_proc, A.d_recvbuf, 1.0, b, sparse_handle, blas_handle);
    if (send.num_msgs)
        RAPtor_MPI_Waitall(send.num_msgs, send.requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    ++A.comm->key;
}
} // namespace raptor
