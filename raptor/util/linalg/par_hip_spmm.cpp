// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_hip_spmm.hpp"
#include "hip_spmm.hpp"
#include "hip_vector.hpp"
#include "raptor/core/comm_pkg.hpp"
#include "raptor/core/hip_error.hpp"
#include <algorithm>
#include <limits>
#include <vector>

namespace raptor
{
namespace
{
bool matches_shape(const ParHipMultiVectorView& v, int global_rows, int local_rows, int cols)
{
    return v.global_rows == global_rows && v.local.rows == local_rows &&
            v.local.cols == cols && cols >= 0 &&
            (local_rows == 0 || cols == 0 || v.local.data != nullptr);
}
}

void par_hip_spmm(double alpha, ParHipCSRMatrix& A, const ParHipMultiVectorView& X,
        double beta, const ParHipMultiVectorView& B, rocsparse_handle sparse_handle,
        rocblas_handle blas_handle)
{
    const int nrhs = X.local.cols;
    if (!matches_shape(X, A.global_num_cols, A.on_proc.n_cols, nrhs) ||
        !matches_shape(B, A.global_num_rows, A.on_proc.n_rows, nrhs))
    {
        std::fprintf(stderr, "Parallel HIP SpMM requires valid views matching the matrix dimensions\n");
        RAPTOR_HIP_ABORT();
        return;
    }
    if (nrhs == 0) return;

    auto& send = *A.comm->send_data;
    auto& recv = *A.comm->recv_data;

    // gaurd against buffer * nrhs overflow
    if (std::max(send.size_msgs, recv.size_msgs) > std::numeric_limits<int>::max() / nrhs)
    {
        std::fprintf(stderr, "Parallel HIP SpMM halo exceeds the supported MPI count\n");
        RAPTOR_HIP_ABORT();
        return;
    }

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));

    // Grow the host and device halo buffers together
    const std::size_t halo_size = static_cast<std::size_t>(recv.size_msgs) * nrhs;
    const std::size_t halo_bytes = halo_size * sizeof(double);
    if (halo_size > A.d_recvbuf_size)
    {
        if (A.d_recvbuf) HIP_CHECK(hipFree(A.d_recvbuf));
        if (A.h_recvbuf) HIP_CHECK(hipHostFree(A.h_recvbuf));
        HIP_CHECK(hipMalloc(&A.d_recvbuf, halo_bytes));
        HIP_CHECK(hipHostMalloc(&A.h_recvbuf, halo_bytes, hipHostMallocDefault));
        A.d_recvbuf_size = halo_size;
    }

    // Posting the nonblocking receive
    recv.recv<double>(A.comm->key, A.comm->mpi_comm, nrhs);
    if (send.num_msgs > 0)
    {
        std::vector<double> h_X(static_cast<std::size_t>(X.local.rows) * nrhs);
        HIP_CHECK(hipMemcpyWithStream(h_X.data(), X.local.data, h_X.size() * sizeof(double),
                hipMemcpyDeviceToHost, stream));

        auto& packed = send.get_buffer<double>();
        packed.resize(static_cast<std::size_t>(send.size_msgs) * nrhs);

        for (int i = 0; i < send.num_msgs; ++i)
        {
            const int start = send.indptr[i];
            const int rows = send.indptr[i + 1] - start;
            double* message = packed.data() + start * nrhs;
            for (int rhs = 0; rhs < nrhs; ++rhs)
            {
                const double* column = h_X.data() + static_cast<std::size_t>(rhs) * X.local.rows;
                for (int row = 0; row < rows; ++row)
                    message[rhs * rows + row] = column[send.indices[start + row]];
            }
            RAPtor_MPI_Isend(message, rows * nrhs, RAPtor_MPI_DOUBLE,
                    send.procs[i], A.comm->key, A.comm->mpi_comm, &send.requests[i]);
        }
    }

    hip_spmm(alpha, A.on_proc, X.local, beta, B.local, sparse_handle, blas_handle);

    recv.waitall();
    if (halo_size > 0)
    {
        const auto& packed = recv.get_buffer<double>();
        for (int i = 0; i < recv.num_msgs; ++i)
        {
            const int start = recv.indptr[i];
            const int rows = recv.indptr[i + 1] - start;
            const double* message = packed.data() + start * nrhs;
            for (int rhs = 0; rhs < nrhs; ++rhs)
                std::copy_n(message + rhs * rows, rows,
                        A.h_recvbuf + rhs * recv.size_msgs + start);
        }
        HIP_CHECK(hipMemcpyWithStream(A.d_recvbuf, A.h_recvbuf, halo_bytes,
                hipMemcpyHostToDevice, stream));
        hip_spmm(alpha, A.off_proc, {A.d_recvbuf, recv.size_msgs, nrhs}, 1.0, B.local,
                sparse_handle, blas_handle);
    }

    send.waitall();
    ++A.comm->key;

    // block CPU till GPU work finishes
    if (halo_size > 0) HIP_CHECK(hipStreamSynchronize(stream));
}

void par_hip_spmm_residual(ParHipCSRMatrix& A, const ParHipMultiVectorView& X,
        const ParHipMultiVectorView& B, const ParHipMultiVectorView& R,
        rocsparse_handle sparse_handle, rocblas_handle blas_handle)
{
    const int nrhs = X.local.cols;
    if (!matches_shape(X, A.global_num_cols, A.on_proc.n_cols, nrhs) ||
        !matches_shape(B, A.global_num_rows, A.on_proc.n_rows, nrhs) ||
        !matches_shape(R, A.global_num_rows, A.on_proc.n_rows, nrhs))
    {
        std::fprintf(stderr, "Parallel HIP SpMM residual requires valid views matching the matrix dimensions\n");
        RAPTOR_HIP_ABORT();
        return;
    }
    if (nrhs == 0) return;

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));
    hip_copy(R.local, B.local, stream);
    par_hip_spmm(-1.0, A, X, 1.0, R, sparse_handle, blas_handle);
}
} // namespace raptor
