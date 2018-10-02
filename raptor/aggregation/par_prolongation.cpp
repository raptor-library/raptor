// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "aggregation/par_prolongation.hpp"

// Assuming weighting = local (not getting approx spectral radius)
ParCSRMatrix* jacobi_prolongation(ParCSRMatrix* A, ParCSRMatrix* T, bool tap_comm,
        double omega, int num_smooth_steps, data_t* comm_t, data_t* comm_mat_t)
{
    ParCSRMatrix* AP_tmp;
    ParCSRMatrix* P_tmp;
    ParCSRMatrix* P = T->copy();
    ParCSRMatrix* scaled_A = A->copy();

    // Get absolute row sum for each row
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    aligned_vector<double> inv_sums;
    if (A->local_num_rows)
    {
        inv_sums.resize(A->local_num_rows, 0);
    }

    double row_sum = 0;
    for (int row = 0; row < A->local_num_rows; row++)
    {
        row_start_on = A->on_proc->idx1[row];
        row_end_on = A->on_proc->idx1[row+1];
        row_sum = 0.0;
        for (int j = row_start_on; j < row_end_on; j++)
        {
            row_sum += fabs(A->on_proc->vals[j]);
        }
        row_start_off = A->off_proc->idx1[row];
        row_end_off = A->off_proc->idx1[row+1];
        for (int j = row_start_off; j < row_end_off; j++)
        {
            row_sum += fabs(A->off_proc->vals[j]);
        }

        if (row_sum)
        {
            inv_sums[row] = (1.0 / fabs(row_sum)) * omega;
        }

        for (int j = row_start_on; j < row_end_on; j++)
        {
            scaled_A->on_proc->vals[j] *= inv_sums[row];
        }
        for (int j = row_start_off; j < row_end_off; j++)
        {
            scaled_A->off_proc->vals[j] *= inv_sums[row];
        }
    }

    // P = P - (scaled_A*P)
    for (int i = 0; i < num_smooth_steps; i++)
    {
        if (tap_comm)
        {
            if (scaled_A->tap_mat_comm == NULL)
            {
                scaled_A->tap_mat_comm = new TAPComm(scaled_A->partition, 
                        scaled_A->off_proc_column_map,
                        scaled_A->on_proc_column_map, 
                        false, MPI_COMM_WORLD, comm_t);
            }
            if (comm_mat_t) *comm_mat_t -= MPI_Wtime();
            AP_tmp = scaled_A->tap_mult(P);
            if (comm_mat_t) *comm_mat_t += MPI_Wtime();
        }
        else
        {
            if (P->comm == NULL)
            {
                scaled_A->comm = new ParComm(scaled_A->partition, 
                        scaled_A->off_proc_column_map,
                        scaled_A->on_proc_column_map, 
                        9283, MPI_COMM_WORLD, comm_t);
            }

            if (comm_mat_t) *comm_mat_t -= MPI_Wtime();
            AP_tmp = scaled_A->mult(P);
            if (comm_mat_t) *comm_mat_t += MPI_Wtime();
        }

        P_tmp = P->subtract(AP_tmp);
        delete AP_tmp;
        delete P;
        P = P_tmp;
        P_tmp = NULL;
    }

    if (tap_comm)
    {
        P->init_tap_communicators();
    }
    else
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map, 
                P->on_proc_column_map, 9283, MPI_COMM_WORLD, comm_t);
    }

    delete scaled_A;
    
    return P;
}

