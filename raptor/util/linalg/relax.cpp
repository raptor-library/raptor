// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "relax.hpp"
#include "spmv.hpp"

/**************************************************************
 *****   Sequential Gauss-Seidel (Forward Sweep)
 **************************************************************
 ***** Performs gauss-seidel along the diagonal block of the 
 ***** matrix, assuming that the off-diagonal block has
 ***** already been altered appropriately.
 ***** The result from this sweep is put into 'result'
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed
 ***** y : data_t*
 *****    Right hand side vector
 ***** result: data_t*
 *****    Vector for result to be put into
 **************************************************************/
void gs_forward(Matrix* A, data_t* x, const data_t* y, data_t* result)
{
    index_t* indptr = A->indptr.data();
    index_t* indices = A->indices.data();
    data_t* data = A->data.data();

    int num_rows = A->n_rows;
    int row_start, row_end;

    int col;
    data_t value;
    data_t diag;

    // Forward Sweep
    for (index_t row = 0; row < num_rows; row++)
    {
        row_start = indptr[row];
        row_end = indptr[row+1];

        if (row_end > row_start) diag = data[row_start];
        else diag = 0.0;

        for (index_t j = row_start + 1; j < row_end; j++)
        {
            col = indices[j];
            value = data[j];
            if (col < row)
            {
                result[row] -= value*result[col];
            }
            else if (col > row)
            {
                result[row] -= value*x[col];
            }
        }
        if (fabs(diag) > zero_tol)
        {
            result[row] /= diag;
        }
    }
}

/**************************************************************
 *****   Sequential Gauss-Seidel (Backward Sweep)
 **************************************************************
 ***** Performs gauss-seidel along the diagonal block of the 
 ***** matrix, assuming that the off-diagonal block has
 ***** already been altered appropriately.
 ***** The result from this sweep is put into 'result'
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed
 ***** y : data_t*
 *****    Right hand side vector
 ***** result: data_t*
 *****    Vector for result to be put into
 **************************************************************/
void gs_backward(Matrix* A, data_t* x, const data_t* y, data_t* result)
{
    index_t* indptr = A->indptr.data();
    index_t* indices = A->indices.data();
    data_t* data = A->data.data();

    int num_rows = A->n_rows;
    int row_start, row_end;
    int col;
    data_t value;
    data_t diag;

    // Backward Sweep
    for (index_t row = num_rows-1; row >= 0; row--)
    {
        row_start = indptr[row];
        row_end = indptr[row+1];

        if (row_end > row_start) diag = data[row_start];
        else diag = 0.0;

        for (index_t j = row_start + 1; j < row_end; j++)
        {
            col = indices[j];
            value = data[j];
            if (col > row)
            {
                result[row] -= value*result[col];
            }
            else if (col < row)
            {
                result[row] -= value*x[col];
            }
        }
        if (fabs(diag) > zero_tol)
        {
            result[row] /= diag;
        }
    }
}

/**************************************************************
 *****   Hybrid Gauss-Seidel / Jacobi Parallel Relaxation
 **************************************************************
 ***** Performs Jacobi along the off-diagonal block and
 ***** symmetric Gauss-Seidel along the diagonal block
 ***** The tmp array is used as a place-holder, but the result
 ***** is returned put in the x-vector. 
 *****
 ***** Parameters
 ***** -------------
 ***** A : Matrix*
 *****    Matrix to relax over
 ***** x : data_t*
 *****    Vector to be relaxed, will contain result
 ***** y : data_t*
 *****    Right hand side vector
 ***** tmp: data_t*
 *****    Vector needed for inner steps
 ***** dist_x : data_t*
 *****    Vector of distant x-values recvd from other processes
 **************************************************************/
void hybrid(ParMatrix* A, data_t* x, const data_t* y, data_t* tmp, data_t* dist_x)
{
    // Forward Sweep
    for (int i = 0; i < A->local_rows; i++)
    {
        tmp[i] = y[i];
    }
    if (A->offd_num_cols)
    {
        sequential_spmv(A->offd, dist_x, tmp, -1.0, 1.0); 
    }
    gs_forward(A->diag, x, y, tmp);

    // Backward Sweep
    for (int i = 0; i < A->local_rows; i++)
    {
        x[i] = y[i];
    }
    if (A->offd_num_cols)
    {
        sequential_spmv(A->offd, dist_x, x, -1.0, 1.0); 
    }
    gs_forward(A->diag, tmp, y, x);
}

/**************************************************************
 *****  Relaxation Method 
 **************************************************************
 ***** Performs jacobi along the diagonal block of the matrix
 *****
 ***** Parameters
 ***** -------------
 ***** l: Level*
 *****    Level in hierarchy to be relaxed
 ***** num_sweeps : int
 *****    Number of relaxation sweeps to perform
 **************************************************************/
void relax(const Level* l, int num_sweeps)
{
    if (l->A->local_rows == 0) return;

    // Get MPI Information
    MPI_Comm comm_mat = l->A->comm_mat;
    int rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    // Declare communication variables
    ParComm*                                 comm;

    // Initialize communication variables
    comm          = l->A->comm;

    data_t* x_data = l->x->local->data();
    data_t* y_data = l->b->local->data();
    data_t* b_tmp_data = l->b_tmp->local->data();
    data_t* x_tmp_data = l->x_tmp->local->data();

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        // If receive values, post appropriate MPI Receives
        if (comm->num_recvs)
        {
            comm->init_recvs(comm_mat);
        }

        // Send values of x to appropriate processors
        if (comm->num_sends)
        {
            comm->init_sends(x_data, comm_mat);
        }

        // Once data is available, add contribution of off-diagonals
        // TODO Deal with new entries as they become available
        // TODO Add an error check on the status
        if (comm->num_recvs)
        {
            // Wait for all receives to finish
            comm->complete_recvs();
        }

        hybrid(l->A, x_data, y_data, x_tmp_data, comm->recv_buffer);

        if (comm->num_sends)
        {
            // Wait for all sends to finish
            // TODO Add an error check on the status
            comm->complete_sends();
        }
    }
}


