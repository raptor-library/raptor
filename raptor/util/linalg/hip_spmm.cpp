// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "hip_spmm.hpp"
#include "hip_vector.hpp"
#include "raptor/core/hip_error.hpp"
#include <stdexcept>

namespace raptor
{
namespace
{
bool matches_shape(const HipMultiVectorView& v, int rows, int cols)
{
    return v.rows == rows && v.cols == cols && cols >= 0 &&
            (rows == 0 || cols == 0 || v.data != nullptr);
}
}

void hip_spmm(double alpha, HipCSRMatrix& A, const HipMultiVectorView& X,
        double beta, const HipMultiVectorView& B, rocsparse_handle sparse_handle,
        rocblas_handle blas_handle)
{
    const int nrhs = X.cols;
    if (!matches_shape(X, A.n_cols, nrhs) || !matches_shape(B, A.n_rows, nrhs))
        throw std::invalid_argument("HIP SpMM requires valid views matching the matrix dimensions");
    if (nrhs == 0 || B.rows == 0) return;

    if (A.nnz == 0)
    {
        if (beta == 1.0) return;

        if (beta == 0.0)
        {
            hipStream_t stream;
            ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));
            const std::size_t bytes = static_cast<std::size_t>(B.rows) * nrhs * sizeof(double);
            HIP_CHECK(hipMemsetAsync(B.data, 0, bytes, stream));
            return;
        }

        ROCBLAS_CHECK(rocblas_dscal_strided_batched(blas_handle, B.rows, &beta,
                B.data, 1, B.rows, nrhs));
        return;
    }

    rocsparse_dnmat_descr mat_b;
    rocsparse_const_dnmat_descr mat_x; // read only
    ROCSPARSE_CHECK(rocsparse_create_const_dnmat_descr(&mat_x,
            X.rows, nrhs, X.rows, X.data,
            rocsparse_datatype_f64_r, rocsparse_order_column));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&mat_b,
            B.rows, nrhs, B.rows, B.data,
            rocsparse_datatype_f64_r, rocsparse_order_column));

    prepare_hip_spmm(A, mat_x, mat_b, nrhs, sparse_handle);
    std::size_t buffer_size = A.spmm_buf_size;
    ROCSPARSE_CHECK(rocsparse_spmm(sparse_handle,
            rocsparse_operation_none, rocsparse_operation_none,
            &alpha, A.descr, mat_x, &beta, mat_b,
            rocsparse_datatype_f64_r, rocsparse_spmm_alg_default,
            rocsparse_spmm_stage_compute, &buffer_size, A.d_spmm_buf));

    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_b));
}

void hip_spmm_residual(HipCSRMatrix& A, const HipMultiVectorView& X,
        const HipMultiVectorView& B, const HipMultiVectorView& R,
        rocsparse_handle sparse_handle, rocblas_handle blas_handle)
{
    const int nrhs = X.cols;
    if (!matches_shape(X, A.n_cols, nrhs) || !matches_shape(B, A.n_rows, nrhs) ||
        !matches_shape(R, A.n_rows, nrhs))
        throw std::invalid_argument("HIP SpMM residual requires valid views matching the matrix dimensions");
    if (nrhs == 0 || R.rows == 0) return;

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));
    hip_copy(R.data, B.data, static_cast<std::size_t>(R.rows) * nrhs, stream);
    hip_spmm(-1.0, A, X, 1.0, R, sparse_handle, blas_handle);
}
} // namespace raptor
