// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* direct_interpolation(const ParCSRMatrix* A,
        const ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states)
{
    int start, end, col;
    int proc, idx, new_idx;
    double sum_strong_pos, sum_strong_neg;
    double sum_all_pos, sum_all_neg;
    double val, alpha, beta, diag;
    double neg_coeff, pos_coeff;

    std::vector<int> on_proc_col_to_new;
    std::vector<int> off_proc_col_to_new;
    if (A->local_num_cols)
    {
        on_proc_col_to_new.resize(A->local_num_cols, -1);
    }
    if (A->off_proc_num_cols)
    {
        off_proc_col_to_new.resize(A->off_proc_num_cols, -1);
    }

    ParCSRMatrix* P = new ParCSRMatrix(A->partition);

    for (int i = 0; i < A->local_num_cols; i++)
    {
        if (states[i])
        {
            on_proc_col_to_new[i] = P->on_proc->col_list.size();
            P->on_proc->col_list.push_back(i);
        }
    }
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (off_proc_states[i])
        {
            off_proc_col_to_new[i] = P->off_proc->col_list.size();
            P->off_proc->col_list.push_back(A->off_proc_column_map[i]);
        }
    }
        
    P->on_proc_column_map = P->on_proc->get_col_list();
    P->off_proc_column_map = P->off_proc->get_col_list();
    
    // Make sure diagonal entry is first in each row of A
    if (!A->on_proc->diag_first)
    {
        A->on_proc->move_diag();
    }

    P->local_num_rows = A->local_num_rows;

    P->local_num_cols = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == 1)
        {
            P->on_proc->idx2.push_back(on_proc_col_to_new[i]);
            P->on_proc->vals.push_back(1);
            P->local_num_cols++;
        }
        else
        {
            sum_strong_pos = 0;
            sum_strong_neg = 0;
            start = S->on_proc->idx1[i];
            end = S->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = S->on_proc->idx2[j]; // Never equals row...
                if (states[col] == 1)
                {
                    val = S->on_proc->vals[j];
                    if (val < 0)
                    {
                        sum_strong_neg += val;
                    }
                    else
                    {
                        sum_strong_pos += val;
                    }
                }
            }
            start = S->off_proc->idx1[i];
            end = S->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = S->off_proc->idx2[j];
                if (off_proc_states[col] == 1)
                {
                    val = S->off_proc->vals[j];
                    if (val < 0)
                    {
                        sum_strong_neg += val;
                    }
                    else
                    {
                        sum_strong_pos += val;
                    }
                }
            }

            sum_all_pos = 0;
            sum_all_neg = 0;
            start = A->on_proc->idx1[i];
            end = A->on_proc->idx1[i+1];
            diag = A->on_proc->vals[start]; // Diag stored first
            for (int j = start+1; j < end; j++)
            {
                col = A->on_proc->idx2[j]; // Never equals row
                if (states[col] == 1)
                {
                    val = A->on_proc->vals[j];
                    if (val < 0)
                    {
                        sum_all_neg += val;
                    }
                    else
                    {
                        sum_all_pos += val;
                    }
                }
            }
            start = A->off_proc->idx1[i];
            end = A->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A->off_proc->idx2[j];
                if (off_proc_states[col] == 1)
                {
                    val = A->off_proc->vals[j];
                    if (val < 0)
                    {
                        sum_all_neg += val;
                    }
                    else
                    {
                        sum_all_pos += val;
                    }
                }
            }

            if (sum_strong_neg == 0)
            {
                alpha = 0;
            }
            else
            {
                alpha = sum_all_neg / sum_strong_neg;
            }

            if (sum_strong_pos == 0)
            {
                diag += sum_all_pos;
                beta = 0;
            }
            else
            {
                beta = sum_all_pos / sum_strong_pos;
            }

            neg_coeff = -alpha / diag;
            pos_coeff = -beta / diag;

            start = S->on_proc->idx1[i];
            end = S->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = S->on_proc->idx2[j];
                if (states[col] == 1)
                {
                    val = S->on_proc->vals[j];
                    P->on_proc->idx2.push_back(on_proc_col_to_new[col]);
                    
                    if (val < 0)
                    {
                        P->on_proc->vals.push_back(neg_coeff * val);
                    }
                    else
                    {
                        P->on_proc->vals.push_back(pos_coeff * val);
                    }
                }
            }
            start = S->off_proc->idx1[i];
            end = S->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = S->off_proc->idx2[j];
                if (off_proc_states[col] == 1)
                {
                    val = S->off_proc->vals[j];
                    P->off_proc->idx2.push_back(off_proc_col_to_new[col]);

                    if (val < 0)
                    {
                        P->off_proc->vals.push_back(neg_coeff * val);
                    }
                    else
                    {
                        P->off_proc->vals.push_back(pos_coeff * val);
                    }
                }
            }
        }
        P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
        P->off_proc->idx1[i+1] = P->off_proc->idx2.size();
    }
    P->on_proc->nnz = P->on_proc->idx2.size();
    P->off_proc->nnz = P->off_proc->idx2.size();
    P->local_nnz = P->on_proc->nnz + P->off_proc->nnz;

    P->off_proc_num_cols = P->off_proc_column_map.size();
    P->on_proc_num_cols = P->on_proc_column_map.size();

    MPI_Allreduce(&(P->local_num_cols), &(P->global_num_cols), 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);

    if (A->comm)
    {
        P->comm = new ParComm(A->comm, on_proc_col_to_new, off_proc_col_to_new);
    }

    if (A->tap_comm)
    {
        P->tap_comm = new TAPComm(A->tap_comm, on_proc_col_to_new,
                off_proc_col_to_new);
    }

    return P;



}




