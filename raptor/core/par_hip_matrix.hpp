// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
// Reference: Communication-Performance-Tutorial/cg-solver-example

#ifndef RAPTOR_CORE_PAR_HIP_MATRIX_HPP
#define RAPTOR_CORE_PAR_HIP_MATRIX_HPP

#include "hip_matrix.hpp"

namespace raptor
{
class ParComm;
class ParCSRMatrix;

/**************************************************************
 *****   Parallel HIP CSR Matrix
 **************************************************************
 ***** This struct stores the device CSR blocks and halo buffers
 ***** needed for distributed GPU SpMV with MPI.
 *****
 ***** Attributes
 ***** -------------
 ***** global_num_rows : int
 *****    Number of rows in the global matrix
 ***** global_num_cols : int
 *****    Number of columns in the global matrix
 ***** on_proc : HipCSRMatrix
 *****    Local matrix block with columns indexing the local vector
 ***** off_proc : HipCSRMatrix
 *****    Local matrix block with columns indexing the ghost vector
 ***** d_sendbuf : double*
 *****    Device buffer containing packed outgoing vector values
 ***** d_recvbuf : double*
 *****    Device buffer containing incoming ghost vector values
 ***** d_send_idx : int*
 *****    Device local-vector indices used to pack the send buffer
 ***** h_sendbuf : double*
 *****    Pinned host send buffer for staged MPI communication
 ***** h_recvbuf : double*
 *****    Pinned host receive buffer for staged MPI communication
 ***** comm : ParComm*
 *****    MPI communicator
 *****
 **************************************************************/
struct ParHipCSRMatrix
{
    int global_num_rows = 0;
    int global_num_cols = 0;

    HipCSRMatrix on_proc;
    HipCSRMatrix off_proc;

    double* d_sendbuf = nullptr;
    double* d_recvbuf = nullptr;
    int* d_send_idx = nullptr;

    double* h_sendbuf = nullptr;
    double* h_recvbuf = nullptr;

    ParComm* comm = nullptr;
};

// Copies CPU matrix to GPU memory
// Creates a rocSPARSE matrix descriptor
void upload_hip_matrix(ParCSRMatrix& host, ParHipCSRMatrix& device,
        rocsparse_handle handle);

// Release both blocks and halo buffers after GPU operations finish
void free_hip_matrix(ParHipCSRMatrix& device);

} // namespace raptor

#endif // RAPTOR_CORE_PAR_HIP_MATRIX_HPP
