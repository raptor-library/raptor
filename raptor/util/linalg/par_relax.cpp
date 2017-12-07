// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "util/linalg/par_relax.hpp"
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
void SOR_forward(ParCSRMatrix* A, ParVector& x, const ParVector& y, 
        const std::vector<double>& dist_x, double omega)
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

void SOR_backward(ParCSRMatrix* A, ParVector& x, const ParVector& y,
        const std::vector<double>& dist_x, double omega)
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
void jacobi(ParLevel* l, int num_sweeps, double omega)
{
    jacobi(l->A, l->x, l->b, l->tmp, num_sweeps, omega);
}
void sor(ParLevel* l, int num_sweeps, double omega)
{
    sor(l->A, l->x, l->b, l->tmp, num_sweeps, omega);
}
void ssor(ParLevel* l, int num_sweeps, double omega)
{
    ssor(l->A, l->x, l->b, l->tmp, num_sweeps, omega);
}

void jacobi(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();
  
    int start, end, col;
    double diag, row_sum;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        A->comm->communicate(x);
        std::vector<double>& dist_x = A->comm->get_recv_buffer<double>();
        for (int i = 0; i < A->local_num_rows; i++)
        {
            tmp[i] = x[i];
        }

        for (int i = 0; i < A->local_num_rows; i++)
        {    
            diag = 0;
            row_sum = 0;

            start = A->on_proc->idx1[i]+1;
            end = A->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A->on_proc->idx2[j];
                row_sum += A->on_proc->vals[j] * tmp[col];
            }

            start = A->off_proc->idx1[i];
            end = A->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A->off_proc->idx2[j];
                row_sum += A->off_proc->vals[j] * dist_x[col];
            }

            if (fabs(diag) > zero_tol)
            {
                x[i] = ((1.0 - omega)*tmp[i]) + (omega*((b[i] - row_sum) / diag));
            }
        }
    }
}

void sor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();
  
    
    

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        A->comm->communicate(x);
        SOR_forward(A, x, b, A->comm->get_recv_buffer<double>(), omega);
    }
}


void ssor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();
  
    
    

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        A->comm->communicate(x);
        SOR_forward(A, x, b, A->comm->get_recv_buffer<double>(), omega);
        SOR_backward(A, x, b, A->comm->get_recv_buffer<double>(), omega);
    }
}


