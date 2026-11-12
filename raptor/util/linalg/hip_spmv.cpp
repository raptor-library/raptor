// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "hip_spmv.hpp"
#include "hip_vector.hpp"
#include "raptor/core/hip_error.hpp"

namespace raptor
{
void hip_residual(HipCSRMatrix& A, const double* x, const double* b,
        double* r, rocsparse_handle handle)
{
    hip_copy(r, b, A.n_rows);
    hip_spmv(-1.0, A, x, 1.0, r, handle);
}

void hip_spmv(double alpha, HipCSRMatrix& A, const double* x,
        double beta, double* b, rocsparse_handle handle)
{
    if (A.nnz == 0) return;

    rocsparse_dnvec_descr vec_x, vec_b;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x, A.n_cols,
            const_cast<double*>(x), rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_b, A.n_rows,
            b, rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none,
            &alpha, A.descr, vec_x, &beta, vec_b,
            rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_compute, &A.spmv_buf_size, A.d_spmv_buf));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_b));
}
} // namespace raptor
