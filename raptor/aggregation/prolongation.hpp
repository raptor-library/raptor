// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_PROLONGATION_HPP
#define RAPTOR_PAR_PROLONGATION_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

using namespace raptor;

ParCSRMatrix* jacobi_prolongation(ParCSRMatrix* A, ParCSRMatrix* T, double omega, 
        int num_smooth_steps)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ParCSRMatrix* AP_tmp;
    ParCSRMatrix* P_tmp;
    ParCSRMatrix* P = new ParCSRMatrix(T);
    P->comm = new ParComm(P->off_proc_column_map, P->first_local_row, P->first_local_col,
            P->global_num_cols, P->local_num_cols);
    ParCSRMatrix* scaled_A = new ParCSRMatrix(A);

    int start, end;
    std::vector<double> inv_sums(A->local_num_rows);
    double row_sum;
     
    for (int row = 0; row < A->local_num_rows; row++)
    {
        row_sum = 0.0;

        start = A->on_proc->idx1[row];
        end = A->on_proc->idx1[row+1];
        for (int j = start; j < end; j++)
        {
            row_sum += fabs(A->on_proc->vals[j]);
        }
        start = A->off_proc->idx1[row];
        end = A->off_proc->idx1[row+1];
        for (int j = start; j < end; j++)
        {
            row_sum += fabs(A->off_proc->vals[j]);
        }

        if (row_sum)
        {
            inv_sums[row] = (1.0 / fabs(row_sum)) * omega;
        }

        for (int j = start; j < end; j++)
        {
            scaled_A->on_proc->vals[j] *= inv_sums[row];
        }
    }

    for (int i = 0; i < num_smooth_steps; i++)
    {
        AP_tmp = scaled_A->mult(P);
        AP_tmp->comm = new ParComm(AP_tmp->off_proc_column_map, AP_tmp->first_local_row,
                AP_tmp->first_local_col, AP_tmp->global_num_cols, AP_tmp->local_num_cols);
        P_tmp = P->subtract(AP_tmp);
        delete AP_tmp;
        delete P;
        P = P_tmp;
        P_tmp = NULL;
    }

    delete scaled_A;

    return P;
}

#endif
