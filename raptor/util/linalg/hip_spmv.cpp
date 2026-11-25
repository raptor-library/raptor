// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "hip_spmv.hpp"
#include "hip_vector.hpp"
#include "raptor/core/hip_error.hpp"

namespace raptor
{
void hip_residual(HipCSRMatrix& A, const double* x, const double* b,
        double* r, rocsparse_handle sparse_handle, rocblas_handle blas_handle)
{
    if (A.n_rows == 0) return;

    hipStream_t stream;
    ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));
    hip_copy(r, b, A.n_rows, stream);
    hip_spmv(-1.0, A, x, 1.0, r, sparse_handle, blas_handle);
}

void hip_spmv(double alpha, HipCSRMatrix& A, const double* x,
        double beta, double* b, rocsparse_handle sparse_handle, rocblas_handle blas_handle)
{
    if (A.n_rows == 0) return;

    if (A.nnz == 0)
    {
        if (beta == 1.0) return;

        if (beta == 0.0)
        {
            hipStream_t stream;
            ROCSPARSE_CHECK(rocsparse_get_stream(sparse_handle, &stream));
            const std::size_t bytes = static_cast<std::size_t>(A.n_rows) * sizeof(double);
            HIP_CHECK(hipMemsetAsync(b, 0, bytes, stream));
            return;
        }

        hip_scale(beta, b, A.n_rows, blas_handle);
        return;
    }

    rocsparse_dnvec_descr vec_b;
    rocsparse_const_dnvec_descr vec_x; // read only
    ROCSPARSE_CHECK(rocsparse_create_const_dnvec_descr(&vec_x, A.n_cols,
            x, rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_b, A.n_rows,
            b, rocsparse_datatype_f64_r));
    prepare_hip_spmv(A, vec_x, vec_b, sparse_handle);
    ROCSPARSE_CHECK(rocsparse_spmv(sparse_handle, rocsparse_operation_none,
            &alpha, A.descr, vec_x, &beta, vec_b,
            rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_compute, &A.spmv_buf_size, A.d_spmv_buf));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_b));
}

} // namespace raptor
