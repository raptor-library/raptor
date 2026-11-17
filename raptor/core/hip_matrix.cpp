// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "hip_matrix.hpp"
#include "hip_error.hpp"
#include "matrix.hpp"

namespace raptor
{
void upload_hip_matrix(const CSRMatrix& host, HipCSRMatrix& device)
{
    device.n_rows = host.n_rows;
    device.n_cols = host.n_cols;
    device.nnz = host.nnz;

    if (host.nnz == 0) return;

    std::size_t row_bytes = (static_cast<std::size_t>(host.n_rows) + 1) * sizeof(int);
    std::size_t index_bytes = static_cast<std::size_t>(host.nnz) * sizeof(int);
    std::size_t value_bytes = static_cast<std::size_t>(host.nnz) * sizeof(double);
    HIP_CHECK(hipMalloc(&device.idx1, row_bytes));
    HIP_CHECK(hipMalloc(&device.idx2, index_bytes));
    HIP_CHECK(hipMalloc(&device.vals, value_bytes));
    HIP_CHECK(hipMemcpy(device.idx1, host.idx1.data(), row_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device.idx2, host.idx2.data(), index_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device.vals, host.vals.data(), value_bytes, hipMemcpyHostToDevice));

    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&device.descr,
            device.n_rows, device.n_cols, device.nnz,
            device.idx1, device.idx2, device.vals,
            rocsparse_indextype_i32, rocsparse_indextype_i32,
            rocsparse_index_base_zero, rocsparse_datatype_f64_r));
}

void prepare_hip_spmv(HipCSRMatrix& device, rocsparse_const_dnvec_descr x,
        rocsparse_dnvec_descr b, rocsparse_handle handle)
{
    if (device.nnz == 0 || device.spmv_prepared) return;

    const double one = 1.0, zero = 0.0;
    ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none,
            &one, device.descr, x, &zero, b,
            rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size, &device.spmv_buf_size, nullptr));
    if (device.spmv_buf_size != 0)
        HIP_CHECK(hipMalloc(&device.d_spmv_buf, device.spmv_buf_size));

    ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none,
            &one, device.descr, x, &zero, b,
            rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_preprocess, &device.spmv_buf_size, device.d_spmv_buf));
    device.spmv_prepared = true;
}

void prepare_hip_spmm(HipCSRMatrix& device, rocsparse_const_dnmat_descr X,
        rocsparse_dnmat_descr B, int nrhs, rocsparse_handle handle)
{
    if (device.nnz == 0 || nrhs == 0 || device.spmm_nrhs == nrhs) return;

    const double one = 1.0, zero = 0.0;
    std::size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_spmm(handle, rocsparse_operation_none,
            rocsparse_operation_none, &one, device.descr, X, &zero, B,
            rocsparse_datatype_f64_r, rocsparse_spmm_alg_default,
            rocsparse_spmm_stage_buffer_size, &buffer_size, nullptr));

    if (buffer_size > device.spmm_buf_size)
    {
        if (device.d_spmm_buf) HIP_CHECK(hipFree(device.d_spmm_buf));
        HIP_CHECK(hipMalloc(&device.d_spmm_buf, buffer_size));
        device.spmm_buf_size = buffer_size;
    }

    ROCSPARSE_CHECK(rocsparse_spmm(handle, rocsparse_operation_none,
            rocsparse_operation_none, &one, device.descr, X, &zero, B,
            rocsparse_datatype_f64_r, rocsparse_spmm_alg_default,
            rocsparse_spmm_stage_preprocess, &buffer_size, device.d_spmm_buf));
    device.spmm_nrhs = nrhs;
}

void free_hip_matrix(HipCSRMatrix& device)
{
    if (device.descr) ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(device.descr));
    if (device.idx1) HIP_CHECK(hipFree(device.idx1));
    if (device.idx2) HIP_CHECK(hipFree(device.idx2));
    if (device.vals) HIP_CHECK(hipFree(device.vals));
    if (device.d_spmv_buf) HIP_CHECK(hipFree(device.d_spmv_buf));
    if (device.d_spmm_buf) HIP_CHECK(hipFree(device.d_spmm_buf));
    device = {};
}
} // namespace raptor
