// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#include "hip_matrix.hpp"
#include "hip_error.hpp"
#include "matrix.hpp"

namespace raptor
{
void upload_hip_matrix(const CSRMatrix& host, HipCSRMatrix& device,
        rocsparse_handle handle)
{
    device.n_rows = host.n_rows;
    device.n_cols = host.n_cols;
    device.nnz = host.nnz;

    if (host.nnz == 0)
    {
        device.idx1 = nullptr;
        device.idx2 = nullptr;
        device.vals = nullptr;
        device.descr = nullptr;
        device.d_spmv_buf = nullptr;
        device.spmv_buf_size = 0;
        return;
    }

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

    double* d_x;
    double* d_b;
    HIP_CHECK(hipMalloc(&d_x, static_cast<std::size_t>(host.n_cols) * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, static_cast<std::size_t>(host.n_rows) * sizeof(double)));
    rocsparse_dnvec_descr vec_x, vec_b;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x, host.n_cols, d_x,
            rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_b, host.n_rows, d_b,
            rocsparse_datatype_f64_r));
    const double one = 1.0, zero = 0.0;
    ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none,
            &one, device.descr, vec_x, &zero, vec_b,
            rocsparse_datatype_f64_r, rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size, &device.spmv_buf_size, nullptr));
    HIP_CHECK(hipMalloc(&device.d_spmv_buf, device.spmv_buf_size));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_b));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
}

void free_hip_matrix(HipCSRMatrix& device)
{
    if (device.nnz == 0) return;
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(device.descr));
    HIP_CHECK(hipFree(device.idx1));
    HIP_CHECK(hipFree(device.idx2));
    HIP_CHECK(hipFree(device.vals));
    HIP_CHECK(hipFree(device.d_spmv_buf));
}
} // namespace raptor
