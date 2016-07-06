// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "spmv.hpp"

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs matrix-vector multiplication on inner indices
 ***** y[inner] = alpha * A[inner, outer] * x[outer] + beta*y[inner]
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void seq_inner_spmv(Matrix* A, const data_t* x, data_t* y, const data_t alpha, const data_t beta, data_t* result = NULL, int outer_start = 0, int n_outer = -1)
{
    index_t* ptr = A->indptr.data();
    index_t* idx = A->indices.data();
    data_t* values = A->data.data();

    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);

    index_t n_inner = A->n_inner;
    if (n_outer < 0)
    {
        n_outer = A->n_outer;
    }
    index_t outer_end = outer_start + n_outer;

    index_t ptr_start;
    index_t ptr_end;

    if (result == NULL)
    {
        result = y;
    }

    if (alpha_one)
    {
        if (beta_zero)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = 0.0;
            }
        }
        else if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
        for (index_t outer = outer_start; outer < outer_end; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                result[inner] += values[j] * x_val;
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
        for (index_t outer = outer_start; outer < outer_end; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                result[inner] -= values[j] * x_val;
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = 0.0;
            }
        }
        else if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
    }
    else
    {
        if (!beta_one)
        {
            for (index_t inner = 0; inner < n_inner; inner++)
            {
                result[inner] = beta * y[inner];
            }
        }
        for (index_t outer = outer_start; outer < outer_end; outer++)
        {
            ptr_start = ptr[outer];
            ptr_end = ptr[outer + 1];
            data_t x_val = x[outer];
            for (index_t j = ptr_start; j < ptr_end; j++)
            {
                index_t inner = idx[j];
                result[inner] += alpha * values[j] * x_val;
            }
        }
    }
}

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs matrix-vector multiplication on outer indices
 ***** y[outer] = alpha * A[outer, inner] * x[inner] + beta*y[outer]
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void seq_outer_spmv(Matrix* A, const data_t* x, data_t* y, const data_t alpha, const data_t beta, data_t* result = NULL, int outer_start = 0, int n_outer = -1)
{
    index_t alpha_zero = (fabs(alpha) < zero_tol);
    index_t alpha_one = (fabs(alpha - 1.0) < zero_tol);
    index_t alpha_neg_one = (fabs(alpha + 1.0) < zero_tol);

    index_t beta_zero = (fabs(beta) < zero_tol);
    index_t beta_one = (fabs(beta - 1.0) < zero_tol);

    index_t* ptr = A->indptr.data();
    index_t* idx = A->indices.data();
    data_t* values = A->data.data();

    if (n_outer <= 0)
    {
        n_outer = A->n_outer;
    }
    index_t outer_end = outer_start + n_outer;

    index_t ptr_start;
    index_t ptr_end;

    if (result == NULL)
    {
        result = y;
    }
    else
    {
        for (int i = 0; i < n_outer; i++)
        {
            result[i] = 0.0;
        }
    }

    if (alpha_one)
    {
        if (beta_one)
        {
            //Ax + y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    result[outer] = values[ptr_start] * x[idx[ptr_start]];
                }
                else
                {
                    result[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += values[j] * x[inner];

                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                result[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_neg_one)
    {
        if (beta_one)
        {
            //-Ax + y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] -= values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //Ax
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    result[outer] = - (values[ptr_start] * x[idx[ptr_start]]);
                }
                else
                {
                    result[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] -= values[j] * x[inner];
                }
            }
        }
        else
        {
            //Ax + beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                result[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] -= values[j] * x[inner];
                }
            }
        }
    }
    else if (alpha_zero)
    {
        if (beta_zero)
        {
            //return 0
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                result[outer] = 0.0;
            }
        }
        else if (!beta_one)
        {
            //beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                result[outer] *= beta;
            }
        }
    }
    else
    {
        if (beta_one)
        {
            //alpha*Ax + y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                for (index_t j = ptr_start; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += alpha * values[j] * x[inner];
                }
            }
        }
        else if (beta_zero)
        {
            //alpha*Ax
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                if (ptr_start < ptr_end)
                {
                    result[outer] = alpha * values[ptr_start] * x[idx[ptr_start]];
                }
                else
                {
                    result[outer] = 0.0;
                }

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += alpha * values[j] * x[inner];
                }
            }
        }
        else
        {
            //alpha * Ax + beta * y
            for (index_t outer = outer_start; outer < outer_end; outer++)
            {
                ptr_start = ptr[outer];
                ptr_end = ptr[outer+1];

                result[outer] *= beta;

                for (index_t j = ptr_start + 1; j < ptr_end; j++)
                {
                    index_t inner = idx[j];
                    result[outer] += alpha * values[j] * x[inner];
                }
            }
        }
    }
}

/**************************************************************
 *****   Sequential Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial matrix-vector multiplication, calling
 ***** method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv(Matrix* A, const data_t* x, data_t* y, const data_t alpha,
    const data_t beta, data_t* result, int outer_start, int outer_end)
{
    if (A->format == CSR)
    {
        seq_outer_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }
    else
    {
        seq_inner_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }   
}

/**************************************************************
 *****   Sequential Transpose Matrix-Vector Multiplication
 **************************************************************
 ***** Performs partial transpose matrix-vector multiplication, 
 ***** calling method appropriate for matrix format
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to be multipled
 ***** x : Vector*
 *****    Vector to be multiplied
 ***** y : Vector*
 *****    Vector result is added to
 ***** alpha : data_t
 *****    Scalar to multipy A*x by
 ***** beta : data_t
 *****    Scalar to multiply original y by
 **************************************************************/
void sequential_spmv_T(Matrix* A, const data_t* x, data_t* y, const data_t alpha, const data_t beta, data_t* result, int outer_start, int outer_end)
{
    if (A->format == CSR)
    {
        seq_inner_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }
    else
    {
        seq_outer_spmv(A, x, y, alpha, beta, result, outer_start, outer_end);
    }   
}

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
        // Compute partial SpMV with local information
        sequential_spmv(A->diag, x_data, y_data, alpha, beta, result_data);
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
        sequential_spmv(A->offd, comm->recv_buffer, y_data, alpha, 
                    1.0, result_data); 
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
void parallel_spmv_T(const ParMatrix* A, const ParVector* x, ParVector* y, const data_t alpha, const data_t beta, const int async, ParVector* result)
{
    // Get MPI Information
    MPI_Comm comm_mat = A->comm_mat;

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

        sequential_spmv_T(A->offd, x_data, send_buffer, alpha, 0.0);
        comm->init_sends_T(comm_mat);
    }

    if (A->local_rows)
    {
        sequential_spmv_T(A->diag, x_data, y_data, alpha, beta);
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

}
