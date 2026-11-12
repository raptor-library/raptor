// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_CORE_HIP_MATRIX_HPP
#define RAPTOR_CORE_HIP_MATRIX_HPP

#include <cstddef>
#include <rocsparse/rocsparse.h>

namespace raptor
{
class CSRMatrix;

/**************************************************************
 *****   HIP CSR Matrix
 **************************************************************
 ***** This struct stores a sparse matrix in device CSR format.
 *****
 ***** Attributes
 ***** -------------
 ***** n_rows : int
 *****    Number of rows
 ***** n_cols : int
 *****    Number of columns
 ***** nnz : int
 *****    Number of nonzeros
 ***** idx1 : int*
 *****    Device row offsets, containing n_rows + 1 entries
 ***** idx2 : int*
 *****    Device column indices, containing nnz entries
 ***** vals : double*
 *****    Device nonzero values, containing nnz entries
 ***** descr : rocsparse_spmat_descr
 *****    rocSPARSE descriptor referencing the device CSR arrays
 ***** d_spmv_buf : void*
 *****    Device workspace reused by rocSPARSE SpMV
 ***** spmv_buf_size : std::size_t
 *****    Size of the SpMV workspace in bytes
 *****
 **************************************************************/
struct HipCSRMatrix
{
    int n_rows = 0;
    int n_cols = 0;
    int nnz = 0;

    int* idx1 = nullptr;
    int* idx2 = nullptr;
    double* vals = nullptr;

    rocsparse_spmat_descr descr = nullptr;

    void* d_spmv_buf = nullptr;
    std::size_t spmv_buf_size = 0;
};

// Copies CPU matrix to GPU memory
// Creates a rocSPARSE matrix descriptor
void upload_hip_matrix(const CSRMatrix& host, HipCSRMatrix& device,
        rocsparse_handle handle);

// Release both blocks and halo buffers after GPU operations finish
void free_hip_matrix(HipCSRMatrix& device);

} // namespace raptor

#endif // RAPTOR_CORE_HIP_MATRIX_HPP
