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
        const aligned_vector<double>& dist_x, double omega)
{
    int start_on, end_on;
    int start_off, end_off;
    int col;
    double diag;
    double row_sum;
    int vec_offset;

    for (int k = 0; k < y.local->b_vecs; k++)
    {
        vec_offset = k * A->local_num_rows;

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
                row_sum += A->on_proc->vals[j] * x[vec_offset + col];
            }
            start_on = end_on;

            end_off = A->off_proc->idx1[i+1];
            for (int j = start_off; j < end_off; j++)
            {
                col = A->off_proc->idx2[j];
                row_sum += A->off_proc->vals[j] * dist_x[vec_offset + col];
            }
            start_off = end_off;

    //        x[i] = ((1.0 - omega)*x[i]) + (omega*((y[i] - row_sum) / diag));
            x[vec_offset + i] = (x[vec_offset + i] + omega * (y[vec_offset + i] - x[vec_offset + i] - row_sum)) / diag;
        }
    }
}

void SOR_backward(ParCSRMatrix* A, ParVector& x, const ParVector& y,
        const aligned_vector<double>& dist_x, double omega)
{
    int start, end, col, vec_offset;
    double diag;
    double row_sum;

    for (int k = 0; k < y.local->b_vecs; k++)
    {
        vec_offset = k * A->local_num_rows; 
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
                row_sum += A->on_proc->vals[j] * x[vec_offset + col];
            }

            start = A->off_proc->idx1[i];
            end = A->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A->off_proc->idx2[j];
                row_sum += A->off_proc->vals[j] * dist_x[vec_offset + col];
            }

            x[vec_offset + i] = ((1.0 - omega)*x[vec_offset + i]) + (omega*((y[vec_offset + i] - row_sum) / diag));
        }
    }
}

void jacobi_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm, data_t* comm_t)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();
  
    int start, end, col, vec_offset;
    double diag, row_sum;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        comm->communicate(x);
        if (comm_t) *comm_t += MPI_Wtime();
        aligned_vector<double>& dist_x = comm->get_buffer<double>();

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

void jacobi_spmv(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm, data_t* comm_t)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();
  
    int start, end, col, vec_offset, dist_vec_offset;
    double diag, row_sum;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        comm->communicate(x, 1);
        if (comm_t) *comm_t += MPI_Wtime();
        aligned_vector<double>& dist_x = comm->get_buffer<double>();

        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        int size = dist_x.size() / x.local->b_vecs;
        double tmp_sum;
        for (int v = 0; v < x.local->b_vecs; v++)
        {
            tmp_sum = 0.0;
            for (int i = 0; i < size; i++)
            {
                tmp_sum += dist_x[v*size + i];
            }
            printf("%d v %d distx_sum %e\n", rank, v, tmp_sum);
        }

        for (int k = 0; k < b.local->b_vecs; k++)
        {
            vec_offset = k * A->local_num_rows;
            dist_vec_offset = dist_x.size() / b.local->b_vecs;
            for (int i = 0; i < A->local_num_rows; i++)
            {
                tmp[vec_offset + i] = x[vec_offset + i];
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
                    row_sum += A->on_proc->vals[j] * tmp[vec_offset + col];
                }
                //printf("%d %d v %d on_proc row_sum %e\n", rank, i, k, row_sum);

                start = A->off_proc->idx1[i];
                end = A->off_proc->idx1[i+1];
                for (int j = start; j < end; j++)
                {
                    col = A->off_proc->idx2[j];
                    row_sum += A->off_proc->vals[j] * dist_x[dist_vec_offset + col];
                }
                //printf("%d %d v %d off_proc row_sum %e\n", rank, i, k, row_sum);

                if (fabs(diag) > zero_tol)
                {
                    x[vec_offset + i] = ((1.0 - omega)*tmp[vec_offset + i]) + (omega*((b[vec_offset + i] - row_sum) / diag));
                }
            }
        }
    }

}

void sor_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm, data_t* comm_t)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        comm->communicate(x);
        if (comm_t) *comm_t += MPI_Wtime();
        SOR_forward(A, x, b, comm->get_buffer<double>(), omega);
    }
}


void ssor_helper(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, CommPkg* comm, data_t* comm_t)
{
    A->on_proc->sort();
    A->off_proc->sort();
    A->on_proc->move_diag();

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        comm->communicate(x);
        if (comm_t) *comm_t += MPI_Wtime();
        SOR_forward(A, x, b, comm->get_buffer<double>(), omega);
        SOR_backward(A, x, b, comm->get_buffer<double>(), omega);
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
void jacobi(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap, data_t* comm_t)
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

    jacobi_spmv(A, x, b, tmp, num_sweeps, omega, comm, comm_t);
    //jacobi_helper(A, x, b, tmp, num_sweeps, omega, comm, comm_t);
}
void sor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap, data_t* comm_t)
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

    sor_helper(A, x, b, tmp, num_sweeps, omega, comm, comm_t);
}
void ssor(ParCSRMatrix* A, ParVector& x, ParVector& b, ParVector& tmp, 
        int num_sweeps, double omega, bool tap, data_t* comm_t)
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

    ssor_helper(A, x, b, tmp, num_sweeps, omega, comm, comm_t);
}







