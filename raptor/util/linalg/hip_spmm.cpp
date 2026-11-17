// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "hip_spmm.hpp"
#include "raptor/core/hip_error.hpp"

namespace raptor
{
void hip_spmm(double alpha, HipCSRMatrix& A, const double* X,
        double beta, double* B, int nrhs, rocsparse_handle handle)
{
    if (nrhs == 0 || A.nnz == 0) return;

    rocsparse_dnmat_descr mat_b;
    rocsparse_const_dnmat_descr mat_x; // read only
    ROCSPARSE_CHECK(rocsparse_create_const_dnmat_descr(&mat_x,
            A.n_cols, nrhs, A.n_cols, X,
            rocsparse_datatype_f64_r, rocsparse_order_column));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&mat_b,
            A.n_rows, nrhs, A.n_rows, B,
            rocsparse_datatype_f64_r, rocsparse_order_column));

    prepare_hip_spmm(A, mat_x, mat_b, nrhs, handle);
    std::size_t buffer_size = A.spmm_buf_size;
    ROCSPARSE_CHECK(rocsparse_spmm(handle,
            rocsparse_operation_none, rocsparse_operation_none,
            &alpha, A.descr, mat_x, &beta, mat_b,
            rocsparse_datatype_f64_r, rocsparse_spmm_alg_default,
            rocsparse_spmm_stage_compute, &buffer_size, A.d_spmm_buf));

    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mat_b));
}

void hip_spmm_residual(HipCSRMatrix& A, const double* X, const double* B,
        double* R, int nrhs, rocsparse_handle handle)
{
    if (nrhs <= 0 || A.n_rows == 0) return;

    hip_copy(R, B, A.n_rows * nrhs);
    hip_spmm(-1.0, A, X, 1.0, R, nrhs, handle);
}
} // namespace raptor
