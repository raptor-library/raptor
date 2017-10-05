// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "util/linalg/relax.hpp"
#include "core/par_matrix.hpp"

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
void gs_forward(ParCSRMatrix* A, ParVector& x, const ParVector& y, 
        const std::vector<double>& dist_x, double omega = 1.0)
{
    int start, end, col;
    double diag;
    double row_sum;

    for (int i = 0; i < A->local_num_rows; i++)
    {
        row_sum = 0;
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        if (A->on_proc->idx2[start] == i)
        {
            diag = A->on_proc->vals[start];
            start++;
        }        
        else continue;
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            row_sum += A->on_proc->vals[j] * x[col];
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            row_sum += A->off_proc->vals[j] * dist_x[col];
        }

        x[i] = ((1.0 - omega)*x[i]) + (omega*((y[i] - row_sum) / diag));
    }
}

void gs_backward(ParCSRMatrix* A, ParVector& x, const ParVector& y,
        const std::vector<double>& dist_x, double omega = 1.0)
{
    int start, end, col;
    double diag;
    double row_sum;

    for (int i = A->local_num_rows - 1; i >= 0; i--)
    {
        row_sum = 0;
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        if (A->on_proc->idx2[start] == i)
        {
            diag = A->on_proc->vals[start];
            start++;
        }        
        else continue;
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            row_sum += A->on_proc->vals[j] * x[col];
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            row_sum += A->off_proc->vals[j] * dist_x[col];
        }

        x[i] = ((1.0 - omega)*x[i]) + (omega*((y[i] - row_sum) / diag));
    }
}

void hybrid(ParCSRMatrix* A, ParVector& x, const ParVector& y, 
       std::vector<double>& dist_x, double omega = 1.0)
{
    gs_forward(A, x, y, dist_x, omega);
    gs_backward(A, x, y, dist_x, omega);
}

void jacobi(ParCSRMatrix* A, ParVector& x, const ParVector& y, ParVector& tmp,
        std::vector<double>& dist_x, double omega = 2.0 / 3)
{
    for (int i = 0; i < A->local_num_rows; i++)
    {
        tmp[i] = x[i];
    }

    for (int i = 0; i < A->local_num_rows; i++)
    { 
        double diag = 0;
        double row_sum = 0;

        int start = A->on_proc->idx1[i];
        int end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            int col = A->on_proc->idx2[j];
            if (col == i)
            {
                diag = A->on_proc->vals[j];
            }
            else
            {
                row_sum += A->on_proc->vals[j] * tmp[col];
            }

        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            int col = A->off_proc->idx2[j];
            row_sum += A->off_proc->vals[j] * dist_x[col];
        }

        if (fabs(diag) > zero_tol)
        {
            x[i] = ((1.0 - omega)*tmp[i]) + (omega*((y[i] - row_sum) / diag));
        }
    }
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
        //hybrid(A, x, b, tmp, A->comm->recv_data->buffer);
        //hybrid(A, x, b, A->comm->recv_data->buffer);
        gs_forward(A, x, b, A->comm->recv_data->buffer, 2.0/3);
    }
}


