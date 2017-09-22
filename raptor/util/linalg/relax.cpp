// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "util/linalg/relax.hpp"
#include "core/par_matrix.hpp"

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
template <typename T, typename U, typename V>
void gs_forward(Matrix* A, const T& x, const U& y, V& result)
{
    int row_start, row_end;
    int col;
    data_t value;
    data_t diag;

    // Forward Sweep
    for (int row = 0; row < A->n_rows; row++)
    {
        row_start = A->idx1[row];
        row_end = A->idx1[row+1];

        if (row_end > row_start) 
        {
            diag = A->vals[row_start];
        }
        else 
        {
            diag = 0.0;
        }

        for (int j = row_start + 1; j < row_end; j++)
        {
            col = A->idx2[j];
            value = A->vals[j];
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
template <typename T, typename U, typename V>
void gs_backward(Matrix* A, const T& x, const U& y, V& result)
{
    int row_start, row_end;
    int col;
    data_t value;
    data_t diag;

    // Backward Sweep
    for (int row = A->n_rows - 1; row >= 0; row--)
    {
        row_start = A->idx1[row];
        row_end = A->idx1[row+1];

        if (row_end > row_start) 
        {
            diag = A->vals[row_start];
        }
        else 
        {
            diag = 0.0;
        }

        for (int j = row_start + 1; j < row_end; j++)
        {
            col = A->idx2[j];
            value = A->vals[j];
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
void hybrid(ParCSRMatrix* A, ParVector& x, const ParVector& y, ParVector& tmp, 
       std::vector<double>& dist_x)
{
    A->sort();

    // Forward Sweep
    if (A->off_proc_num_cols)
    {
        A->off_proc->residual(dist_x, y.local, tmp.local);
    }
    gs_forward(A->on_proc, x, y, tmp);

    // Backward Sweep
    if (A->off_proc_num_cols)
    {
        A->off_proc->residual(dist_x, y.local, x.local);
    }
    gs_forward(A->on_proc, tmp, y, x);
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
void relax(Level* l, int num_sweeps)
{
    relax(l->A, l->x, l->b, l->tmp, num_sweeps);
}

void relax(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, int num_sweeps)
{
    if (A->local_num_rows == 0)
    {
        return;
    }
    if (!A->on_proc->sorted)
    {
        A->on_proc->sort();
    }
    if (!A->off_proc->sorted)
    {
        A->off_proc->sort();
    }
    if (!A->on_proc->diag_first)
    {
        A->on_proc->move_diag();
    }
  
    for (int i = 0; i < num_sweeps; i++)
    {
        // Send / Recv X-Data
        A->comm->communicate(x);

        // Run hybrid jacobi / gauss seidel
        hybrid(A, x, b, tmp, A->comm->recv_data->buffer);
    }
}


