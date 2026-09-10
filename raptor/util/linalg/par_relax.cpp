// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "raptor/core/types.hpp"
#include "par_relax.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/util/linalg/lapack_wrapper.hpp"

#ifdef USING_OPENMP
#include <omp.h>
#endif

namespace raptor {
// Declare Private Methods
void SOR_forward(ParCSRMatrix* A, ParVector& x, const ParVector& y, 
        const std::vector<double>& dist_x, double omega);
void SOR_backward(ParCSRMatrix* A, ParVector& x, const ParVector& y,
        const std::vector<double>& dist_x, double omega);
void jacobi_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm);
void sor_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm);
void ssor_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm);



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
    int start_on, end_on;
    int start_off, end_off;
    int col;
    double diag;
    double row_sum;

    start_on = 0;
    start_off = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        row_sum = 0;
        end_on = A->on_proc->idx1[i+1];
        if (A->on_proc->idx2[start_on] == i)
        {
            diag = A->on_proc->vals[start_on];
            start_on++;
        }        
        else continue;
        for (int j = start_on; j < end_on; j++)
        {
            col = A->on_proc->idx2[j];
            row_sum += A->on_proc->vals[j] * x[col];
        }
        start_on = end_on;

        end_off = A->off_proc->idx1[i+1];
        for (int j = start_off; j < end_off; j++)
        {
            col = A->off_proc->idx2[j];
            row_sum += A->off_proc->vals[j] * dist_x[col];
        }
        start_off = end_off;

//        x[i] = ((1.0 - omega)*x[i]) + (omega*((y[i] - row_sum) / diag));
        x[i] = (x[i] + omega * (y[i] - x[i] - row_sum)) / diag;
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

void jacobi_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();
  
    for (int iter = 0; iter < num_sweeps; iter++)
    {
        comm->communicate(x);
        std::vector<double>& dist_x = comm->get_buffer<double>();
        for (int i = 0; i < A->local_num_rows; i++)
        {
            tmp[i] = x[i];
        }

        #ifdef USING_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < A->local_num_rows; i++)
        {    
            double row_sum = 0;

      
            int start = A->on_proc->idx1[i];
            const int end = A->on_proc->idx1[i+1];
            if (start == end || A->on_proc->idx2[start] != i)
                continue;

            const double diag = A->on_proc->vals[start++];

            for (int j = start; j < end; j++)
            {
                const int col = A->on_proc->idx2[j];
                row_sum += A->on_proc->vals[j] * tmp[col];
            }

            const int start_off = A->off_proc->idx1[i];
            const int end_off = A->off_proc->idx1[i+1];
            for (int j = start_off; j < end_off; j++)
            {
                const int col = A->off_proc->idx2[j];
                row_sum += A->off_proc->vals[j] * dist_x[col];
            }

            if (fabs(diag) > zero_tol)
            {
                x[i] = ((1.0 - omega)*tmp[i]) + (omega*((b[i] - row_sum) / diag));
            }
        }
    }
}

void sor_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        comm->communicate(x);
        SOR_forward(A, x, b, comm->get_buffer<double>(), omega);
    }
}


void ssor_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        comm->communicate(x);
        SOR_forward(A, x, b, comm->get_buffer<double>(), omega);
        SOR_backward(A, x, b, comm->get_buffer<double>(), omega);
    }
}

static void block_jacobi_helper(ParBSRMatrix* A, const std::vector<double>& block_diag_inv, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega)
{
    const auto bsize = A->on_proc->b_size;
    const auto brow = A->on_proc->b_rows;
    const auto bcol = A->on_proc->b_cols;
  
    assert(brow == bcol);

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        A->residual(x,b,tmp,false);

        #ifdef USING_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < A->local_num_rows; i++)
        {    
            int offset = i * brow;
            const double* D_inv = block_diag_inv.data() + i*bsize;

            for (int br = 0; br < brow; br++)
            {
                double correction = 0.0;

                for (int bc = 0; bc < bcol; bc++)
                {
                    correction += D_inv[br * bcol + bc]* tmp[offset + bc];
                }

                x[offset + br] += omega * correction;
            }
        }
    }
}

std::vector<double> compute_block_diag_inv(const ParBSRMatrix* A)
{
    // A is assumed sorted and move_diag() already in each level
    BSRMatrix* A_on = static_cast<BSRMatrix*>(A->on_proc);

    assert(A_on->b_rows == A_on->b_cols);

    int row_start, row_end;

    std::vector<double> diag_inv(A->local_num_rows * A_on -> b_size, 0.0);
    for (int row = 0; row < A-> local_num_rows; row ++)
    {
        row_start = A_on->idx1[row];
        row_end = A_on->idx1[row+1];

        // empty row or no diagonal block that row
        if (row_start == row_end || A_on -> idx2[row_start] != row)
        {
            // filled with 0s
            continue;
        }
        
        double* diag_block = diag_inv.data() + row * A_on->b_size;

        std::copy(A_on->block_vals[row_start],A_on->block_vals[row_start] + A_on->b_size, diag_block);
    }

    for (int row = 0; row < A-> local_num_rows; row ++)
    {
        double* block = diag_inv.data() + row*A_on->b_size;
        auto pinv_block = pinv(block, A_on->b_rows, A_on->b_cols, -1.0);
        std::copy(pinv_block.begin(), pinv_block.end(), block);
    }

    return diag_inv;
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
void jacobi(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap)
{
    CommPkg* comm;
    if (tap)
    {
        if (!A->tap_comm) 
        {
            A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        comm = A->tap_comm;
    }
    else
    {
        if (!A->comm) 
        {
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        comm = A->comm;
    }

    jacobi_helper(A, x, b, tmp, num_sweeps, omega, comm);
}
void sor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap)
{
    CommPkg* comm;
    if (tap)
    {
        if (!A->tap_comm) 
        {
            A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        comm = A->tap_comm;
    }
    else
    {
        if (!A->comm) 
        {
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        comm = A->comm;
    }

    sor_helper(A, x, b, tmp, num_sweeps, omega, comm);
}
void ssor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap)
{
    CommPkg* comm;
    if (tap)
    {
        if (!A->tap_comm) 
        {
            A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        comm = A->tap_comm;
    }
    else
    {
        if (!A->comm) 
        {
            A->comm = new ParComm(A->partition, A->off_proc_column_map,
                    A->on_proc_column_map);
        }
        comm = A->comm;
    }

    ssor_helper(A, x, b, tmp, num_sweeps, omega, comm);
}
void block_jacobi(ParBSRMatrix* A, const std::vector<double>& block_diag_inv, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega)
{
    block_jacobi_helper(A, block_diag_inv, x, b, tmp, num_sweeps, omega);
}

}
