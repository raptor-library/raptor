// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states)
{
    int start, end;
    int start_k, end_k;
    int end_S;
    int col, col_k, col_S;
    int ctr;
    int global_col;
    
    
    

    double diag, val;
    double weak_sum, coarse_sum;
    
    double sign;

    CSRMatrix* recv_mat; // Communicate A
    CSRMatrix* recv_on; // On Proc Block of Recvd A
    CSRMatrix* recv_off; // Off Proc Block of Recvd A

    A->sort();
    S->sort();
    A->on_proc->move_diag();
    S->on_proc->move_diag();

    // Initialize P
    std::vector<int> on_proc_col_to_new;
    std::vector<int> off_proc_col_to_new;
    std::vector<bool> col_exists;
    if (S->on_proc_num_cols)
    {
        on_proc_col_to_new.resize(S->on_proc_num_cols, -1);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_col_to_new.resize(S->off_proc_num_cols, -1);
        col_exists.resize(S->off_proc_num_cols, false);
    }

    int off_proc_cols = 0;
    int on_proc_cols = 0;
    int global_num_cols;
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == 1)
        {
            on_proc_cols++;
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == 1)
        {
            off_proc_cols++;
        }
    }
    MPI_Allreduce(&(on_proc_cols), &(global_num_cols), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
    ParCSRMatrix* P = new ParCSRMatrix(A->partition, A->global_num_rows, global_num_cols, 
            A->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (states[i] == 1)
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.push_back(S->on_proc_column_map[i]);
        }
    }
    P->local_row_map = S->get_local_row_map();

    // Need off_proc_states_A
    std::vector<int> off_proc_states_A;
    if (A->off_proc_num_cols) off_proc_states_A.resize(A->off_proc_num_cols);
    A->comm->communicate(states);
    for (int i = 0; i < A->comm->recv_data->size_msgs; i++)
    {
        off_proc_states_A[i] = A->comm->recv_data->int_buffer[i];
    } 

    // Map off proc cols S to A
    std::vector<int> off_proc_S_to_A;
    if (S->off_proc_num_cols) off_proc_S_to_A.resize(S->off_proc_num_cols);
    ctr = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        while (S->off_proc_column_map[i] != A->off_proc_column_map[ctr])
        {
            ctr++;
        }
        off_proc_S_to_A[i] = ctr;
    }

    // Communicate parallel matrix A (Costly!)
    recv_mat = A->comm->communicate(A);

    // Add all coarse columns of S to global_to_local (map)
    ctr = 0;
    std::map<int, int> global_to_local;
    for (std::vector<int>::iterator it = A->off_proc_column_map.begin();
            it != A->off_proc_column_map.end(); ++it)
    {
        global_to_local[*it] = ctr++;
    }
    // Break up recv_mat into recv_on and recv_off
    int* on_proc_partition_to_col = A->map_partition_to_local();
    recv_on = new CSRMatrix(A->off_proc_num_cols, A->on_proc_num_cols);
    recv_off = new CSRMatrix(A->off_proc_num_cols, A->off_proc_num_cols);
    recv_on->idx1[0] = 0;
    recv_off->idx1[0] = 0;
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = recv_mat->idx2[j];

            if (global_col >= A->partition->first_local_row
                    && global_col <= A->partition->last_local_row)
            {
                recv_on->idx2.push_back(on_proc_partition_to_col[
                            global_col - A->partition->first_local_row]);
                recv_on->vals.push_back(recv_mat->vals[j]);
            }
            // In off_proc_column_map
            else 
            {
                std::map<int, int>::iterator it = global_to_local.find(global_col);
                if (it != global_to_local.end())
                {
                    recv_off->idx2.push_back(it->second);
                    recv_off->vals.push_back(recv_mat->vals[j]);
                }
            }
        }
        recv_on->idx1[i+1] = recv_on->idx2.size();
        recv_off->idx1[i+1] = recv_off->idx2.size();
    }
    recv_on->nnz = recv_on->idx2.size();
    recv_off->nnz = recv_off->idx2.size();
    delete[] on_proc_partition_to_col;
    delete recv_mat;

    // For each row, will calculate coarse sums and store 
    // strong connections in vector
    std::vector<int> pos;
    std::vector<int> off_proc_pos;
    std::vector<int> row_coarse;
    std::vector<int> off_proc_row_coarse;
    std::vector<double> row_strong;
    std::vector<double> off_proc_row_strong;
    if (A->on_proc_num_cols)
    {
        pos.resize(A->on_proc_num_cols, -1);
        row_coarse.resize(A->on_proc_num_cols, 0);
        row_strong.resize(A->on_proc_num_cols, 0);
    }
    if (A->off_proc_num_cols)
    {
        off_proc_pos.resize(A->off_proc_num_cols, -1);
        off_proc_row_coarse.resize(A->off_proc_num_cols, 0);
        off_proc_row_strong.resize(A->off_proc_num_cols, 0);
    }

    for (int i = 0; i < A->local_num_rows; i++)
    {
        // If coarse row, add to P
        if (states[i] == 1)
        {
            P->on_proc->idx2.push_back(on_proc_col_to_new[i]);
            P->on_proc->vals.push_back(1);
            P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
            P->off_proc->idx1[i+1] = P->off_proc->idx2.size();
            continue;
        }
        // Store diagonal value
        start = A->on_proc->idx1[i] + 1;
        end = A->on_proc->idx1[i+1];
        diag = A->on_proc->vals[start-1];

        if (diag < 0)
        {
            sign = -1.0;
        }
        else
        {
            sign = 1.0;
        }

        // Determine weak sum for row and store coarse / strong columns
        ctr = S->on_proc->idx1[i] + 1;
        end_S = S->on_proc->idx1[i+1];
        weak_sum = diag;
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            if (ctr < end_S && S->on_proc->idx2[ctr] == col)
            {
                if (states[col] == 1)
                {
                    pos[col] = P->on_proc->idx2.size();
                    P->on_proc->idx2.push_back(on_proc_col_to_new[col]);
                    P->on_proc->vals.push_back(val);
                }

                if (states[col] != -3)
                {
                    row_coarse[col] = states[col];
                    row_strong[col] = (1 - states[col]) * val;
                }
                ctr++;
            }
            else // weak connection
            {
                weak_sum += val;
            }
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        ctr = S->off_proc->idx1[i];
        end_S = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            val = A->off_proc->vals[j];
            global_col = A->off_proc_column_map[col];
            
            // If strong connection
            if (ctr < end_S && S->off_proc_column_map[S->off_proc->idx2[ctr]] == global_col)
            {
                col_S = S->off_proc->idx2[ctr];
                if (off_proc_states_A[col] == 1)
                {
                    off_proc_pos[col] = P->off_proc->idx2.size();
                    col_exists[col_S] = true;
                    P->off_proc->idx2.push_back(col_S);
                    P->off_proc->vals.push_back(val);
                }
                if (off_proc_states_A[col] != -3)
                {
                    off_proc_row_coarse[col] = off_proc_states_A[col];
                    off_proc_row_strong[col] = (1 - off_proc_states_A[col]) * val;
                }
                ctr++;
            }
            else
            {
                weak_sum += val;
            }
        }

        start = A->on_proc->idx1[i] + 1;
        end = A->on_proc->idx1[i+1];
        ctr = S->on_proc->idx1[i] + 1;
        end_S = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j]; // k
            if (ctr < end_S && S->on_proc->idx2[ctr] == col)
            {
                if (states[col] == 0) // Not coarse: k in D_i^s
                {
                    // Find sum of all coarse points in row k (with sign NOT equal to diag)
                    coarse_sum = 0;
                    start_k = A->on_proc->idx1[col] + 1;
                    end_k = A->on_proc->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = A->on_proc->idx2[k];  // m
                        val = A->on_proc->vals[k] * row_coarse[col_k];
                        if (val * sign < 0)
                        {
                            coarse_sum += val;
                        }
                    }
                    start_k = A->off_proc->idx1[col];
                    end_k = A->off_proc->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = A->off_proc->idx2[k]; // m
                        val = A->off_proc->vals[k] * off_proc_row_coarse[col_k];
                        if (val * sign < 0)
                        {
                            coarse_sum += val;
                        }
                    }
                    if (fabs(coarse_sum) < zero_tol)
                    {
                        weak_sum += A->on_proc->vals[j];
                        row_strong[col] = 0;
                    }
                    else
                    {
                        row_strong[col] /= coarse_sum;  // holds val for a_ik/sum_C(a_km)
                    }
                }
                ctr++;
            }
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        ctr = S->off_proc->idx1[i];
        end_S = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            if (ctr < end_S && S->off_proc_column_map[S->off_proc->idx2[ctr]] 
                    == A->off_proc_column_map[col])
            {
                if (off_proc_states_A[col] == 0) // Not Coarse
                {
                    // Strong connection... create 
                    coarse_sum = 0;
                    start_k = recv_on->idx1[col];
                    end_k = recv_on->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = recv_on->idx2[k];
                        val = recv_on->vals[k] * row_coarse[col_k];
                        if (val * sign < 0)
                        {
                            coarse_sum += val;
                        }
                    }
                    start_k = recv_off->idx1[col];
                    end_k = recv_off->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = recv_off->idx2[k];
                        val = recv_off->vals[k] * off_proc_row_coarse[col_k];
                        if (val * sign < 0)
                        {
                            coarse_sum += val;
                        }
                    }
                    if (fabs(coarse_sum) < zero_tol)
                    {
                        weak_sum += A->off_proc->vals[j];
                        off_proc_row_strong[col] = 0;
                    }
                    else
                    {
                        off_proc_row_strong[col] /= coarse_sum; // holds val for a_ik/sum_C(a_km)
                    }
                }
                ctr++;
            }
        }

        int idx;
        // Find weight for each Sij
        start = S->on_proc->idx1[i] + 1;
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j]; // k
            if (states[col] == 0) // k in D_i^S
            {
                start_k = A->on_proc->idx1[col]+1;
                end_k = A->on_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->on_proc->idx2[k];
                    val = A->on_proc->vals[k];
                    idx = pos[col_k];
                    if (val * sign < 0 && idx >= 0)
                    {
                        P->on_proc->vals[idx] += (row_strong[col] * val);
                    }
                }

                start_k = A->off_proc->idx1[col];
                end_k = A->off_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->off_proc->idx2[k];
                    val = A->off_proc->vals[k];
                    idx = off_proc_pos[col_k];
                    if (val * sign < 0 && idx >= 0)
                    {
                        P->off_proc->vals[idx] += (row_strong[col] * val);
                    }
                }
            }
        }

        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col_S = S->off_proc->idx2[j];
            col = off_proc_S_to_A[col_S];
            if (off_proc_states_A[col] == 0)
            {
                start_k = recv_on->idx1[col];
                end_k = recv_on->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    val = recv_on->vals[k];
                    col_k = recv_on->idx2[k];
                    idx = pos[col_k];
                    if (val * sign < 0 && idx >= 0)
                    {
                        P->on_proc->vals[idx] += (off_proc_row_strong[col] * val);
                    }
                }

                start_k = recv_off->idx1[col];
                end_k = recv_off->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    val = recv_off->vals[k];
                    col_k = recv_off->idx2[k];
                    idx = off_proc_pos[col_k];
                    if (val * sign < 0 && idx >= 0)
                    {
                        P->off_proc->vals[idx] += (off_proc_row_strong[col] * val);
                    }
                }
            }
        }

        start = S->on_proc->idx1[i]+1;
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            idx = pos[col];
            if (states[col] == 1)
            {
                P->on_proc->vals[idx] /= -weak_sum;
            }
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = off_proc_S_to_A[S->off_proc->idx2[j]];
            idx = off_proc_pos[col];
            if (off_proc_states_A[col] == 1)
            {
                P->off_proc->vals[idx] /= -weak_sum;
            }
        }



        // Clear row values
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            pos[col] = -1;
            row_coarse[col] = 0;
            row_strong[col] = 0;
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = off_proc_S_to_A[S->off_proc->idx2[j]];
            off_proc_pos[col] = -1;
            off_proc_row_coarse[col] = 0;
            off_proc_row_strong[col] = 0;
        }

        P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
        P->off_proc->idx1[i+1] = P->off_proc->idx2.size();
    }
    P->on_proc->nnz = P->on_proc->idx2.size();
    P->off_proc->nnz = P->off_proc->idx2.size();
    P->local_nnz = P->on_proc->nnz + P->off_proc->nnz;

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            off_proc_col_to_new[i] = P->off_proc_column_map.size();
            P->off_proc_column_map.push_back(S->off_proc_column_map[i]);
        }
    }
    for (std::vector<int>::iterator it = P->off_proc->idx2.begin(); 
            it != P->off_proc->idx2.end(); ++it)
    {
        *it = off_proc_col_to_new[*it];
    }


    P->off_proc_num_cols = P->off_proc_column_map.size();
    P->on_proc_num_cols = P->on_proc_column_map.size();
    P->off_proc->n_cols = P->off_proc_num_cols;
    P->on_proc->n_cols = P->on_proc_num_cols;

    if (S->comm)
    {
        P->comm = new ParComm(S->comm, on_proc_col_to_new, off_proc_col_to_new);
    }

    if (A->tap_comm)
    {
        P->tap_comm = new TAPComm(A->tap_comm, on_proc_col_to_new,
                off_proc_col_to_new);
    }

    delete recv_on;
    delete recv_off;

    return P;
}


ParCSRMatrix* direct_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states)
{
    int start, end, col;
    
    int ctr;
    double sum_strong_pos, sum_strong_neg;
    double sum_all_pos, sum_all_neg;
    double val, alpha, beta, diag;
    double neg_coeff, pos_coeff;

    A->sort();
    S->sort();
    A->on_proc->move_diag();
    S->on_proc->move_diag();

    // Copy entries of A into sparsity pattern of S
    std::vector<double> sa_on;
    std::vector<double> sa_off;
    if (S->on_proc->nnz)
    {
        sa_on.resize(S->on_proc->nnz);
    }
    if (S->off_proc->nnz)
    {
        sa_off.resize(S->off_proc->nnz);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        ctr = A->on_proc->idx1[i];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc_column_map[S->on_proc->idx2[j]];
            while (A->on_proc_column_map[A->on_proc->idx2[ctr]] != col)
            {
                ctr++;
            }
            sa_on[j] = A->on_proc->vals[ctr];
        }

        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        ctr = A->off_proc->idx1[i];
        for (int j = start; j < end; j++)
        {
            col = S->off_proc_column_map[S->off_proc->idx2[j]];
            while (A->off_proc_column_map[A->off_proc->idx2[ctr]] != col)
            {
                ctr++;
            }
            sa_off[j] = A->off_proc->vals[ctr];
        }
    }

    std::vector<int> on_proc_col_to_new;
    std::vector<int> off_proc_col_to_new;
    if (S->on_proc_num_cols)
    {
        on_proc_col_to_new.resize(S->on_proc_num_cols, -1);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_col_to_new.resize(S->off_proc_num_cols, -1);
    }

    int off_proc_cols = 0;
    int on_proc_cols = 0;
    int global_num_cols;
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i])
        {
            on_proc_cols++;
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i])
        {
            off_proc_cols++;
        }
    }
    MPI_Allreduce(&(on_proc_cols), &(global_num_cols), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
    ParCSRMatrix* P = new ParCSRMatrix(S->partition, S->global_num_rows, global_num_cols, 
            S->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i])
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.push_back(S->on_proc_column_map[i]);
        }
    }
    std::vector<bool> col_exists;
    if (S->off_proc_num_cols)
    {
        col_exists.resize(S->off_proc_num_cols, false);
    }
    P->local_row_map = S->get_local_row_map();

    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == 1)
        {
            P->on_proc->idx2.push_back(on_proc_col_to_new[i]);
            P->on_proc->vals.push_back(1);
        }
        else
        {
            sum_strong_pos = 0;
            sum_strong_neg = 0;
            sum_all_pos = 0;
            sum_all_neg = 0;

            start = S->on_proc->idx1[i];
            end = S->on_proc->idx1[i+1];
            if (S->on_proc->idx2[start] == i)
            {
                start++;
            }
            for (int j = start; j < end; j++)
            {
                col = S->on_proc->idx2[j]; 
                if (states[col] == 1)
                {
                    val = sa_on[j];
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
                    val = sa_off[j];
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

            start = A->on_proc->idx1[i];
            end = A->on_proc->idx1[i+1];
            diag = A->on_proc->vals[start]; // Diag stored first
            start++;
            for (int j = start; j < end; j++)
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
            start = A->off_proc->idx1[i];
            end = A->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
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

            alpha = sum_all_neg / sum_strong_neg;
           

            //if (sum_strong_neg == 0)
            //{
            //    alpha = 0;
            //}
            //else
            //{
            //    alpha = sum_all_neg / sum_strong_neg;
            //}

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
            if (S->on_proc->idx2[start] == i)
            {
                start++;
            }
            for (int j = start; j < end; j++)
            {
                col = S->on_proc->idx2[j];
                if (states[col] == 1)
                {
                    val = sa_on[j];
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
                    val = sa_off[j];
                    col_exists[col] = true;
                    P->off_proc->idx2.push_back(col);

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
    
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            off_proc_col_to_new[i] = P->off_proc_column_map.size();
            P->off_proc_column_map.push_back(S->off_proc_column_map[i]);
        }
    }
    for (std::vector<int>::iterator it = P->off_proc->idx2.begin(); 
            it != P->off_proc->idx2.end(); ++it)
    {
        *it = off_proc_col_to_new[*it];
    }


    P->off_proc_num_cols = P->off_proc_column_map.size();
    P->on_proc_num_cols = P->on_proc_column_map.size();
    P->off_proc->n_cols = P->off_proc_num_cols;
    P->on_proc->n_cols = P->on_proc_num_cols;

    if (S->comm)
    {
        P->comm = new ParComm(S->comm, on_proc_col_to_new, off_proc_col_to_new);
    }

    if (A->tap_comm)
    {
        P->tap_comm = new TAPComm(A->tap_comm, on_proc_col_to_new,
                off_proc_col_to_new);
    }

    return P;
}





