// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "spmv.hpp"

/**************************************************************
 *****   Parallel Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel matrix-vector multiplication
 ***** y = alpha*A*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
void parallel_spmv(const ParMatrix* A, const ParVector* x, ParVector* y, const data_t alpha, const data_t beta, const int async, ParVector* result)
{
    data_t* result_data = NULL;
    if (result != NULL && result->local_n)
    {
        result_data = result->local->data();
    }

    // Get MPI Information
    MPI_Comm comm_mat = A->comm_mat;

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    data_t*                                  x_data;
    data_t*                                  y_data;
    index_t                                  ctr;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    ParComm*                                 comm;

    // Initialize communication variables
    comm          = A->comm;
    num_sends = comm->num_sends;
    num_recvs = comm->num_recvs;

    if (x->local_n)
    {
        x_data = x->local->data();
    }

    if (y->local_n)
    {
        y_data = y->local->data();
    }

    if (num_recvs)
    {
        comm->init_recvs(comm_mat);
    }
    if (num_sends)
    {
        comm->init_sends(x_data, comm_mat);
    }

    if (A->local_rows)
    {
        A->diag->mult(x_data, y_data);
    }

    // Once data is available, add contribution of off-diagonals
    // TODO Deal with new entries as they become available
    // TODO Add an error check on the status
    if (num_recvs)
    {
        // TODO Add an error check on the status
        // Wait for all receives to finish
        comm->complete_recvs();

        // Add received data to Vector
        A->offd->mult(comm->recv_buffer, y_data);
    }

    // TODO Add an error check on the status
    if (num_sends)
    {
        // Wait for all sends to finish
        comm->complete_sends();
    }
 }

/**************************************************************
 *****   Parallel Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs parallel transpose matrix-vector multiplication
 ***** y = alpha*A^T*x + beta*y
 *****
 ***** Parameters
 ***** -------------
 ***** A : ParMatrix*
 *****    Parallel matrix to be transposed and multipled
 ***** x : ParVector*
 *****    Parallel vector to be multiplied
 ***** y : ParVector*
 *****    Parallel vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy alpha*A*x
 ***** beta : data_t
 *****    Scalar to multiply original y by
 ***** async : index_t
 *****    Boolean flag for updating SpMV asynchronously
 **************************************************************/
// TODO -- Make this work with new matrix class
void parallel_spmv_T(const ParMatrix* A, const ParVector* x, ParVector* y, const data_t alpha, const data_t beta, const int async, ParVector* result)
{
    // Get MPI Information
/*    MPI_Comm comm_mat = A->comm_mat;

    // TODO must enforce that y and x are not aliased, or this will NOT work
    //    or we could use blocking sends as long as we post the iRecvs first

    // Declare communication variables
    data_t*                                  send_buffer;
    data_t*                                  recv_buffer;
    data_t*                                  x_data;
    data_t*                                  y_data;
    index_t                                  begin;
    index_t                                  ctr;
    index_t                                  request_ctr;
    index_t                                  tag;
    index_t                                  num_sends;
    index_t                                  num_recvs;
    index_t*                                 recv_col_starts;
    index_t*                                 recv_col_indices;
    ParComm*                                 comm;

    // Initialize communication variables
    comm          = A->comm;
    num_sends = comm->num_recvs;
    num_recvs = comm->num_sends;

    if (num_recvs)
    {
        recv_col_starts = comm->send_row_starts.data();
        recv_col_indices = comm->send_row_indices.data();
        recv_buffer = comm->send_buffer;
    }

    if (x->local_n)
    {
        x_data = x->local->data();
    }

    if (y->local_n)
    {
        y_data = y->local->data();
    }

    if (num_recvs)
    {
        comm->init_recvs_T(comm_mat);
    }

    // Perform local spmv -- into send_buffer
    // Send this result
    if (num_sends)
    {
        send_buffer = comm->recv_buffer;

        index_t proc, num_send, send_start, send_end;

        //sequential_spmv_T(A->offd, x_data, send_buffer, alpha, 0.0);
        comm->init_sends_T(comm_mat);
    }

    if (A->local_rows)
    {
        //sequential_spmv_T(A->diag, x_data, y_data, alpha, beta);
    }

    if (num_recvs)
    {
        int recv_idx, recv_start, recv_end, num_cols;
        comm->complete_recvs_T();

        for (int i = 0; i < num_recvs; i++)
        {
            recv_start = recv_col_starts[i];
            recv_end = recv_col_starts[i+1];
            for (int j = recv_start; j < recv_end; j++)
            {
                index_t row = recv_col_indices[j];
                y_data[row] += recv_buffer[j];
            }
        }
    }

    if (num_sends)
    {
        // Wait for all sends to finish
        comm->complete_sends_T();
    }
*/
}
