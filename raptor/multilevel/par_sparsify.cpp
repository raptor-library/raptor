// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "multilevel/par_sparsify.hpp"

using namespace raptor;

void sparsify(ParCSRMatrix* A, ParCSRMatrix* P, ParCSRMatrix* I, 
        ParCSRMatrix* AP, ParCSRMatrix* Ac, const double theta)
{
    // Form Minimal Sparsity Pattern
    ParCSRMatrix* M1 = AP->mult_T(I);
    ParCSRMatrix* AI = A->mult(I);
    ParCSRMatrix* M2 = AI->mult_T(P);
    ParCSRMatrix* M = M1->add(M2);
    delete AI;
    delete M1;
    delete M2;

    int nnz;
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Reduce(&M->local_nnz, &nnz, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, 0, RAPtor_MPI_COMM_WORLD);
    if (rank == 0) printf("NNZ %d\n", nnz);

    int diag_pos, ctr_on, ctr_off;
    int start_on, start_off;
    int end_on, end_off;
    int ctr_M, end_M;
    int col_A, global_col_A;
    int col_M, global_col_M;
    double max_val, val;
    double val_A;
    double diag;

    std::vector<int> off_proc_col_exists;

    Ac->sort();
    M->sort();
    Ac->on_proc->move_diag();
    M->on_proc->move_diag();
    if (Ac->off_proc_num_cols)
        off_proc_col_exists.resize(Ac->off_proc_num_cols, 0);

    // Go through each row of Ac... If not in M and smaller than rel tol, remove
    diag_pos = 0;
    ctr_on = 0;
    ctr_off = 0;
    start_on = Ac->on_proc->idx1[0];
    start_off = Ac->off_proc->idx1[0];
    for (int i = 0; i < Ac->local_num_rows; i++)
    {
        // Find abs max val in row (off diag)
        max_val = 0.0;
        diag = Ac->on_proc->vals[start_on++];

        end_on = Ac->on_proc->idx1[i+1];
        end_off = Ac->off_proc->idx1[i+1];

        // Max val in row
        for (int j = start_on; j < end_on; j++)
        {
            val = fabs(Ac->on_proc->vals[j]);
            if (val > max_val)
            {
                max_val = val;
            }
        }

        for (int j = start_off; j < end_off; j++)
        {
            val = fabs(Ac->off_proc->vals[j]);
            if (val > max_val)
            {
                max_val = val;
            }
        }

        // Add diagonal
        diag_pos = ctr_on;
        Ac->on_proc->idx2[ctr_on] = i;
        Ac->on_proc->vals[ctr_on++] = diag;

        // For each val in row, check if in M, or if greater than theta*row_max
        ctr_M = M->on_proc->idx1[i];
        end_M = M->on_proc->idx1[i+1];
        if (M->on_proc->idx2[ctr_M] == i)
        {
            ctr_M++;
        }
        for (int j = start_on; j < end_on; j++)
        {
            col_A = Ac->on_proc->idx2[j];
            val_A = Ac->on_proc->vals[j];
            if (ctr_M < end_M)
            {
                col_M = M->on_proc->idx2[ctr_M];
                while (ctr_M + 1 < end_M && col_M < col_A)
                {
                    col_M = M->on_proc->idx2[++ctr_M];
                }
            }

            // Keep in A
            if ((ctr_M < end_M && col_A == col_M) || fabs(val_A) >= theta * max_val)          
            {   
                Ac->on_proc->idx2[ctr_on] = col_A;
                Ac->on_proc->vals[ctr_on++] = val_A;
            }
            else // Add to diagonal (remove)
            {
                Ac->on_proc->vals[diag_pos] += val_A;
            }
        }
        start_on = end_on;
        Ac->on_proc->idx1[i+1] = ctr_on;

        ctr_M = M->off_proc->idx1[i];
        end_M = M->off_proc->idx1[i+1];
        for (int j = start_off; j < end_off; j++)
        {
            col_A = Ac->off_proc->idx2[j];
            global_col_A = Ac->off_proc_column_map[col_A];
            val_A = Ac->off_proc->vals[j];
            if (ctr_M < end_M)
            {
                col_M = M->off_proc->idx2[ctr_M];
                global_col_M = M->off_proc_column_map[col_M];
                while (ctr_M + 1 < end_M && global_col_M < global_col_A)
                {
                    col_M = M->off_proc->idx2[++ctr_M];
                    global_col_M = M->off_proc_column_map[col_M];
                }
            }

            // Keep in A
            if ((ctr_M < end_M && global_col_A == global_col_M) || fabs(val_A) >= theta * max_val)
            {
                Ac->off_proc->idx2[ctr_off] = col_A;
                Ac->off_proc->vals[ctr_off++] = val_A;
                off_proc_col_exists[col_A] = 1;
            }
            else // Add to diagonal (remove)
            {
                Ac->on_proc->vals[diag_pos] += val_A;
            }
        }
        start_off = end_off;
        Ac->off_proc->idx1[i+1] = ctr_off; 
    }

    Ac->on_proc->nnz = ctr_on;
    Ac->off_proc->nnz = ctr_off;
    Ac->local_nnz = ctr_on + ctr_off;

    std::vector<int> off_proc_col_to_new;
    if (Ac->off_proc_num_cols)
    {
        off_proc_col_to_new.resize(Ac->off_proc_num_cols);
    }
    int ctr = 0;
    for (int i = 0; i < Ac->off_proc_num_cols; i++)
    {
        if (off_proc_col_exists[i])
        {
            off_proc_col_to_new[i] = ctr;
            Ac->off_proc_column_map[ctr++] = Ac->off_proc_column_map[i];
        }
    }
    Ac->off_proc_column_map.resize(ctr);
    Ac->off_proc_num_cols = ctr;

    for (std::vector<int>::iterator it = Ac->off_proc->idx2.begin();
            it != Ac->off_proc->idx2.end(); ++it)
    {
        *it = off_proc_col_to_new[*it];
    }

    // Update communicate package (only recv if off proc col exists)
    if (Ac->comm)
    {
        Ac->comm->update(off_proc_col_exists);
    }

    delete M;
}
