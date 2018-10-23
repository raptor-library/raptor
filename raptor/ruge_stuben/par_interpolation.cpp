// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

// TODO -- if in S, col is positive, otherwise col is -(col+1)
CSRMatrix* communicate(ParCSRMatrix* A, ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states, CommPkg* comm)
{
    int start, end, col;
    int ctr_S, end_S, global_col;
    int tmp_col, sign;
    double diag, val;

    aligned_vector<int> rowptr(A->local_num_rows + 1);
    aligned_vector<int> col_indices;
    aligned_vector<double> values;
    if (A->local_nnz)
    {
        col_indices.reserve(A->local_nnz);
        values.reserve(A->local_nnz);
    }
    rowptr[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        ctr_S = S->on_proc->idx1[i]+1;
        end_S = S->on_proc->idx1[i+1];

        diag = A->on_proc->vals[start++];
        if (diag > 0) sign = 1;
        else sign = -1;

        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            if (states[col] == Selected)
            {
                global_col = A->on_proc_column_map[col];
                if (ctr_S < end_S && S->on_proc->idx2[ctr_S] == col)
                {
                    if (val * sign < 0) // Only add needed cols
                    {
                        col_indices.emplace_back(-(global_col+1));
                        values.emplace_back(val);
                    }
                    ctr_S++;
                }
                else
                {
                    if (val * sign < 0) // Only add needed cols
                    {
                        col_indices.emplace_back(global_col);
                        values.emplace_back(val);
                    }
                }
            }
            else if (ctr_S < end_S && S->on_proc->idx2[ctr_S] == col)
            {
                ctr_S++;
            }
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        ctr_S = S->off_proc->idx1[i];
        end_S = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            val = A->off_proc->vals[j];
            if (off_proc_states[col] == NoNeighbors) continue;

            global_col = A->off_proc_column_map[col];

            if (off_proc_states[col] == Unselected) // Add for +i possibility
            {
                if (val * sign < 0) // Only add needed cols
                {
                    global_col += A->partition->global_num_cols;
                    col_indices.emplace_back(global_col);
                    values.emplace_back(val);
                }
                if (ctr_S < end_S && S->off_proc->idx2[ctr_S] == col)
                {
                    ctr_S++;
                }
            }
            else if (ctr_S < end_S && S->off_proc->idx2[ctr_S] == col) 
            {
                // Selected and in S
                if (val * sign < 0) //Only add needed cols 
                {
                    col_indices.emplace_back(-(global_col+1));
                    values.emplace_back(val);
                }
                ctr_S++;
            }
            else
            {
                // Selected weak connection
                if (val * sign < 0)
                {
                    col_indices.emplace_back(global_col);
                    values.emplace_back(val);
                }
            }
        }
        rowptr[i+1] = col_indices.size();
    }

    return comm->communicate(rowptr, col_indices, values);
   
}


CSRMatrix*  communicate(ParCSRMatrix* A, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states, CommPkg* comm)
{
    int start, end, col;
    int sign;
    double val;

    aligned_vector<int> rowptr(A->local_num_rows + 1);
    aligned_vector<int> col_indices;
    aligned_vector<double> values;
    if (A->local_nnz)
    {
        col_indices.reserve(A->local_nnz);
        values.reserve(A->local_nnz);
    }

    rowptr[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        val = A->on_proc->vals[start++];
        if (val > 0) sign = 1;
        else sign = -1;

        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            if (states[col] == 1 && val * sign < 0)
            {
                col_indices.emplace_back(A->on_proc_column_map[col]);
                values.emplace_back(A->on_proc->vals[j]);
            }
        }
        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->off_proc->idx2[j];
            val = A->off_proc->vals[j];
            if (off_proc_states[col] == 1 && val * sign < 0)
            {
                col_indices.emplace_back(A->off_proc_column_map[col]);
                values.emplace_back(A->off_proc->vals[j]);
            }
        }
        rowptr[i+1] = col_indices.size();
    }

    return comm->communicate(rowptr, col_indices, values);
}

ParCSRMatrix* extended_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states, 
        bool tap_interp, int num_variables, int* variables, 
        data_t* comm_t, data_t* comm_mat_t)
{
    int start, end, idx, idx_k;
    int ctr, end_S;
    int start_k, end_k;
    int col, global_col;
    int col_k, col_P;
    int global_num_cols;
    int on_proc_cols, off_proc_cols;
    int sign;
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    double val, val_k, weak_sum;
    double diag, col_sum;

    CommPkg* comm = A->comm;
    CommPkg* mat_comm = A->comm;
    if (tap_interp)
    {
        comm = A->tap_comm;
        mat_comm = A->tap_mat_comm;
    }

    std::set<int> global_set;
    std::map<int, int> global_to_local;
    aligned_vector<int> off_proc_column_map;
    aligned_vector<int> off_variables;

    CSRMatrix* recv_mat; // On Proc Block of Recvd A

    // If not already sorted, sort A and S together (move diag first)
    A->sort();
    S->sort();
    A->on_proc->move_diag();
    S->on_proc->move_diag();

    // Communicate off_variables if num_variables > 1
    if (A->off_proc_num_cols) off_variables.resize(A->off_proc_num_cols);
    if (num_variables > 1)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        comm->communicate(variables);
        if (comm_t) *comm_t += MPI_Wtime();
        
        for (int i = 0; i < A->off_proc_num_cols; i++)
        {
            off_variables[i] = comm->get_int_buffer()[i];
        }
    }

    // Communicate parallel matrix A (portion needed)
    if (comm_mat_t) *comm_mat_t -= MPI_Wtime();
    recv_mat = communicate(A, S, states, off_proc_states, mat_comm);
    if (comm_mat_t) *comm_mat_t += MPI_Wtime();

    int tmp_col = col;
    int* on_proc_partition_to_col = A->map_partition_to_local();
    
    aligned_vector<int> A_recv_on_ptr(recv_mat->n_rows + 1);
    aligned_vector<int> S_recv_on_ptr(recv_mat->n_rows + 1);
    aligned_vector<int> A_recv_off_ptr(recv_mat->n_rows + 1);
    aligned_vector<int> S_recv_off_ptr(recv_mat->n_rows + 1);
    aligned_vector<int> A_recv_on_idx(recv_mat->nnz);
    aligned_vector<int> S_recv_on_idx(recv_mat->nnz);
    aligned_vector<int> A_recv_off_idx(recv_mat->nnz);
    aligned_vector<int> S_recv_off_idx(recv_mat->nnz);

    A_recv_on_ptr[0] = 0;
    S_recv_on_ptr[0] = 0;
    A_recv_off_ptr[0] = 0;
    S_recv_off_ptr[0] = 0;

    int A_recv_on_ctr = 0;
    int S_recv_on_ctr = 0;
    int A_recv_off_ctr = 0;
    int S_recv_off_ctr = 0;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = recv_mat->idx2[j];
            val = recv_mat->vals[j];

            tmp_col = col;
            if (col < 0) 
            {
                col = (-col) - 1;
            }
            if (col >= A->partition->global_num_cols)
            {
                col -= A->partition->global_num_cols;
                if (col >= A->partition->first_local_col &&
                        col <= A->partition->last_local_col)
                {
                    // Only add to A (for +i)
                    col = on_proc_partition_to_col[col - A->partition->first_local_row];
                    A_recv_on_idx[A_recv_on_ctr++] = j;
                }
            }
            // Otherwise, add every value to A (and neg values also to S)
            else if (col < A->partition->first_local_col || 
                    col > A->partition->last_local_col)
            {
                if (tmp_col < 0) // Only add to S if neg
                {
                    S_recv_off_idx[S_recv_off_ctr++] = j;
                }
                A_recv_off_idx[A_recv_off_ctr++] = j;
            }
            else
            {
                col = on_proc_partition_to_col[col - A->partition->first_local_row];
                if (tmp_col < 0) // Only add to S if neg
                {
                    S_recv_on_idx[S_recv_on_ctr++] = j;
                }
                A_recv_on_idx[A_recv_on_ctr++] = j;
            }
            recv_mat->idx2[j] = col;
        }
        A_recv_on_ptr[i+1] = A_recv_on_ctr;
        S_recv_on_ptr[i+1] = S_recv_on_ctr;
        A_recv_off_ptr[i+1] = A_recv_off_ctr;
        S_recv_off_ptr[i+1] = S_recv_off_ctr;
    }
    A_recv_on_idx.resize(A_recv_on_ctr);
    A_recv_on_idx.shrink_to_fit();
    S_recv_on_idx.resize(S_recv_on_ctr);
    S_recv_on_idx.shrink_to_fit();
    A_recv_off_idx.resize(A_recv_off_ctr);
    A_recv_off_idx.shrink_to_fit();
    S_recv_off_idx.resize(S_recv_off_ctr);
    S_recv_off_idx.shrink_to_fit();

    delete[] on_proc_partition_to_col;

    // Change off_proc_cols to local (remove cols not on rank)
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == Selected)
        {
            global_set.insert(S->off_proc_column_map[i]);
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == Unselected)
        {
            start = A_recv_off_ptr[i];
            end = A_recv_off_ptr[i+1];
            for (int j = start; j < end; j++)
            {
                global_set.insert(recv_mat->idx2[A_recv_off_idx[j]]);
            }
        }
    }
    for (std::set<int>::iterator it = global_set.begin(); it != global_set.end(); ++it)
    {
        global_to_local[*it] = off_proc_column_map.size();
        off_proc_column_map.emplace_back(*it);
    }
    off_proc_cols = off_proc_column_map.size();

    for (aligned_vector<int>::iterator it = A_recv_off_idx.begin(); 
            it != A_recv_off_idx.end(); ++it)
    {
        recv_mat->idx2[*it] = global_to_local[recv_mat->idx2[*it]];
    }

    // Initialize P
    aligned_vector<int> on_proc_col_to_new;
    aligned_vector<bool> col_exists;
    if (S->on_proc_num_cols)
    {
        on_proc_col_to_new.resize(S->on_proc_num_cols, -1);
    }
    if (off_proc_cols)
    {
        col_exists.resize(off_proc_cols, false);
    }
    on_proc_cols = 0;
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == Selected)
        {
            on_proc_cols++;
        }
    }
    // Initialize AllReduce to determine global num cols
    MPI_Request reduce_request;
    int reduce_buf = on_proc_cols;
    MPI_Iallreduce(&(reduce_buf), &global_num_cols, 1, MPI_INT, MPI_SUM, 
            MPI_COMM_WORLD, &reduce_request);
   
    ParCSRMatrix* P = new ParCSRMatrix(A->partition, A->global_num_rows, -1, 
            A->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == Selected)
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.emplace_back(S->on_proc_column_map[i]);
        }
    }
    P->local_row_map = S->get_local_row_map();

    aligned_vector<int> off_proc_A_to_P;
    if (A->off_proc_num_cols) 
    {
	    off_proc_A_to_P.resize(A->off_proc_num_cols, -1);
    }
    ctr = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] != Selected)
        {
            continue; // Only for coarse points
        }

        while (off_proc_column_map[ctr] < A->off_proc_column_map[i])
        {
            ctr++;
        }
        off_proc_A_to_P[i] = ctr;
    }

    // For each row, will calculate coarse sums and store 
    // strong connections in vector
    aligned_vector<int> pos;
    aligned_vector<int> off_proc_pos;
    aligned_vector<double> coarse_sum;
    aligned_vector<double> off_proc_coarse_sum;
    if (A->on_proc_num_cols)
    {
        pos.resize(A->on_proc_num_cols, -1);
        coarse_sum.resize(A->on_proc_num_cols);
    }
    if (A->off_proc_num_cols)
    {
        off_proc_coarse_sum.resize(A->off_proc_num_cols);
    }
    if (P->off_proc_num_cols)
    {
        off_proc_pos.resize(P->off_proc_num_cols, -1);
    }


    // Find upperbound size of P->on_proc and P->off_proc
    int nnz_on = 0;
    int nnz_off = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == Unselected)
        {
            nnz_on += (S->on_proc->idx1[i+1] - S->on_proc->idx1[i]);
            nnz_off += (S->off_proc->idx1[i+1] - S->off_proc->idx1[i]);

            start = S->on_proc->idx1[i]+1;
            end = S->on_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = S->on_proc->idx2[j];
                if (states[col] == Unselected)
                {
                    nnz_on += (S->on_proc->idx1[col+1] - S->on_proc->idx1[col]);
                    nnz_off += (S->off_proc->idx1[col+1] - S->off_proc->idx1[col]);
                }
            }
            start = S->off_proc->idx1[i];
            end = S->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = S->off_proc->idx2[j];
                if (off_proc_states[col] == Unselected)
                {
                    nnz_on += (S_recv_on_ptr[col+1] - S_recv_on_ptr[col]);
                    nnz_off += (S_recv_off_ptr[col+1] - S_recv_off_ptr[col]);
                }
            }
        }
        else
        {
            nnz_on++;
        }
    }
    P->on_proc->idx2.resize(nnz_on);
    P->on_proc->vals.resize(nnz_on);
    P->off_proc->idx2.resize(nnz_off);
    P->off_proc->vals.resize(nnz_off);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    nnz_on = 0;
    nnz_off = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        // If coarse row, add to P
        if (states[i] != Unselected)
        {
            if (states[i] == Selected)
            {
                P->on_proc->idx2[nnz_on] = on_proc_col_to_new[i];
                P->on_proc->vals[nnz_on++] = 1;
            }
            P->on_proc->idx1[i+1] = nnz_on;
            P->off_proc->idx1[i+1] = nnz_off;
            continue;
        }

        // Go through strong coarse points, 
        // add to row coarse and create sparsity of P (dist1)
        row_start_on = nnz_on;
        row_start_off = nnz_off;

        start = S->on_proc->idx1[i]+1;
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            val = S->on_proc->vals[j];
            if (states[col] == Selected)
            {
                if (pos[col] < row_start_on)
                {
                    pos[col] = nnz_on;
                    P->on_proc->idx2[nnz_on] = on_proc_col_to_new[col];
                    P->on_proc->vals[nnz_on++] = 0.0;
                }
            }
            else if (states[col] == Unselected)
            {
                start_k = S->on_proc->idx1[col]+1;
                end_k = S->on_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = S->on_proc->idx2[k];
                    if (states[col_k] == Selected && pos[col_k] < row_start_on)
                    {
                        pos[col_k] = nnz_on;
                        P->on_proc->idx2[nnz_on] = on_proc_col_to_new[col_k];
                        P->on_proc->vals[nnz_on++] = 0.0;
                    }
                }

                start_k = S->off_proc->idx1[col];
                end_k = S->off_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = S->off_proc->idx2[k];
                    col_P = off_proc_A_to_P[col_k];
                    if (off_proc_states[col_k] == Selected && off_proc_pos[col_P] < row_start_off)
                    {
                        col_exists[col_P] = true;
                        off_proc_pos[col_P] = nnz_off;
                        P->off_proc->idx2[nnz_off] = col_P;
                        P->off_proc->vals[nnz_off++] = 0.0;
                    }
                }
            }
        }

        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];
            val = S->off_proc->vals[j];
            if (off_proc_states[col] == Selected)
            {
                col_P = off_proc_A_to_P[col];
                if (off_proc_pos[col_P] < row_start_off)
                {
                    off_proc_pos[col_P] = nnz_off;
                    col_exists[col_P] = true;
                    P->off_proc->idx2[nnz_off] = col_P;
                    P->off_proc->vals[nnz_off++] = 0.0;
                }
            }
            else if (off_proc_states[col] == Unselected)
            {
                start_k = S_recv_on_ptr[col];
                end_k = S_recv_on_ptr[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    idx = S_recv_on_idx[k];
                    col_k = recv_mat->idx2[idx];
                    if (pos[col_k] < row_start_on)
                    {
                        pos[col_k] = nnz_on;
                        P->on_proc->idx2[nnz_on] = on_proc_col_to_new[col_k];
                        P->on_proc->vals[nnz_on++] = 0.0;
                    }
                }

                start_k = S_recv_off_ptr[col];
                end_k = S_recv_off_ptr[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    idx = S_recv_off_idx[k];
                    col_k = recv_mat->idx2[idx];
                    if (off_proc_pos[col_k] < row_start_off)
                    {
                        off_proc_pos[col_k] = nnz_off;
                        P->off_proc->idx2[nnz_off] = col_k;
                        P->off_proc->vals[nnz_off++] = 0.0;
                        col_exists[col_k] = true;
                    }
                }
            }
        }
        pos[i] = nnz_on;
        row_end_on = nnz_on;
        row_end_off = nnz_off;


        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        weak_sum = A->on_proc->vals[start++]; // Add a_ii to weak sum
if (rank == 15 && i == 0) printf("Weak Sum = %e\n", weak_sum);
        ctr = S->on_proc->idx1[i]+1;
        end_S = S->on_proc->idx1[i+1];

        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            idx = pos[col];
            if (idx >= row_start_on)
            {
                P->on_proc->vals[idx] += val;
                if (ctr < end_S && S->on_proc->idx2[ctr] == col)
                    ctr++;
            }
            else if (ctr < end_S && S->on_proc->idx2[ctr] == col)
            {
                ctr++;

                if (states[col] != Unselected) continue;
                
                // sum a_kl for k in F-points of S and l in C^_i U {i}
                // k = col (unselected, in S)
                col_sum = 0;

                // Find sum of all coarse points in row k (with sign NOT equal to diag)
                start_k = A->on_proc->idx1[col];
                end_k = A->on_proc->idx1[col+1];

                // Only add a_kl if sign(a_kl) != sign (a_kk)
                diag = A->on_proc->vals[start_k++];
                if (diag > 0) sign = 1;
                else sign = -1;

                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->on_proc->idx2[k]; // a_kl
                    val_k = A->on_proc->vals[k];

                    // sign(a_kl) != sign(a_kk) and a_kl in row of P (or == i)
                    if (val_k * sign < 0 && pos[col_k] >= row_start_on)
                    {
                        col_sum += val_k;
                    }
                }

                start_k = A->off_proc->idx1[col];
                end_k = A->off_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->off_proc->idx2[k]; // a_kl
                    val_k = A->off_proc->vals[k];
                    col_P = off_proc_A_to_P[col_k];
                    // sign(a_kl) != sign(a_kk) and a_kl in row of P
                    if (col_P >= 0 && val_k * sign < 0 && off_proc_pos[col_P] >= row_start_off)
                    {
                        col_sum += val_k;
                    }
                }

                // If no strong connections (col_sum == 0), add to weak_sum
                if (fabs(col_sum) < zero_tol)
                {
                    weak_sum += val;
if (rank == 15 && i == 0) printf("1. Weak Sum += %e\n", val);
                }
                else // Otherwise, add products to P
                {
                    col_sum = val / col_sum;  // product = a_ik / col_sum

                    start_k = A->on_proc->idx1[col]+1; 
                    end_k = A->on_proc->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = A->on_proc->idx2[k]; // a_kj for some j
                        val_k = A->on_proc->vals[k];
                        idx = pos[col_k]; // Find idx of w_ij for j^^
                        // if sign(a_kj) != sign(a_kk) and j in C^_{i} U {i}
                        if (val_k * sign < 0 && idx >= row_start_on)
                        {
                            if (col_k == i) // if j == i, add to weak sum
                            {
                                weak_sum += (col_sum * val_k);
if (rank == 15 && i == 0) printf("2. Weak Sum += %e\n", col_sum * val_k);
                            }
                            else // Otherwise, add to w_ij
                            {
                                P->on_proc->vals[idx] += (col_sum * val_k);
                            }
                        }
                    }
                
                    start_k = A->off_proc->idx1[col];
                    end_k = A->off_proc->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = A->off_proc->idx2[k]; // a_kj for some j
                        col_P = off_proc_A_to_P[col_k];
                        if (col_P >= 0) // If column not in P, col not in C^_{i}
                        {
                            val_k = A->off_proc->vals[k];
                            idx = off_proc_pos[col_P]; // Find idx of w_ij 
                            // If sign(a_kj) != sign(a_kk) and j in C^_{i}
                            if (val_k * sign < 0 && idx >= row_start_off)
                            {
                                // Add to w_ij
                                P->off_proc->vals[idx] += (col_sum * val_k);
                            }
                        }
                    } 
                }
            }
            else // Weak connection, add to weak_sum if not in C^_{i}
            {
                if (num_variables == 1 || variables[i] == variables[col])// weak connection
                {
                    if (states[col] != NoNeighbors)
                    {
                        weak_sum += val;
if (rank == 15 && i == 0) printf("3. Weak Sum += %e\n", val);
                    }
                }
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
            idx = off_proc_pos[off_proc_A_to_P[col]];
            if (idx >= row_start_off)
            {
                P->off_proc->vals[idx] += val;
                if (ctr < end_S && S->off_proc->idx2[ctr] == col)
                    ctr++;
            }
            else if (ctr < end_S && S->off_proc->idx2[ctr] == col)
            {
                ctr++;

                if (off_proc_states[col] != Unselected) continue;

                col_sum = 0;

                // Add recvd values not in S
                start_k = A_recv_on_ptr[col];
                end_k = A_recv_on_ptr[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    idx = A_recv_on_idx[k];
                    col_k = recv_mat->idx2[idx];
                    val_k = recv_mat->vals[idx];
                    if (pos[col_k] >= row_start_on) // Checked val * sign before communication
                    {
                        col_sum += val_k;
                    }
                }

                start_k = A_recv_off_ptr[col];
                end_k = A_recv_off_ptr[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    idx = A_recv_off_idx[k];
                    col_k = recv_mat->idx2[idx];
                    val_k = recv_mat->vals[idx];
                    if (off_proc_pos[col_k] >= row_start_off) // Checked val * sign before communication
                    {
                        col_sum += val_k;
                    }
                }

                if (fabs(col_sum) < zero_tol)
                {
                    weak_sum += val;
if (rank == 15 && i == 0) printf("4. Weak Sum += %e\n", val);
                }
                else
                {
                    col_sum = val / col_sum;

                    start_k = A_recv_on_ptr[col];
                    end_k = A_recv_on_ptr[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        idx = A_recv_on_idx[k];
                        col_k = recv_mat->idx2[idx];
                        val_k = recv_mat->vals[idx];
                        idx = pos[col_k];
                        if (idx >= row_start_on) // Checked val * sign before communication
                        {
                            if (col_k == i)
                            {
                                weak_sum += (col_sum * val_k);
if (rank == 15 && i == 0) printf("5. Weak Sum += %e\n", col_sum * val_k);
                            }
                            else
                            {
                                P->on_proc->vals[idx] += (col_sum * val_k);
                            }
                        }
                    }

                    start_k = A_recv_off_ptr[col];
                    end_k = A_recv_off_ptr[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        idx = A_recv_off_idx[k];
                        col_k = recv_mat->idx2[idx];
                        val_k = recv_mat->vals[idx];
                        idx = off_proc_pos[col_k];
                        if (idx >= row_start_off) // Checked val * sign before communication
                        {
                            P->off_proc->vals[idx] += (col_sum * val_k);
                        }
                    }
                }
            }
            else // Weak connection, add to weak sum if not in C^_{i}
            {
                if (num_variables == 1 || variables[i] == off_variables[col])
                {
                    if (off_proc_states[col] != NoNeighbors)
                    {
if (rank == 15 && i == 0) printf("6. Weak Sum += %e\n", val);
                        weak_sum += val;
                    }
                }
            }
        }

if (rank == 15 && i == 0) printf("Weak Sum %e\n", weak_sum);
        // Divide by weak sum and clear row values
        if (fabs(weak_sum) > zero_tol)
        {
            for (int j = row_start_on; j < row_end_on; j++)
            {
                P->on_proc->vals[j] /= -weak_sum;
            }
            for (int j = row_start_off; j < row_end_off; j++)
            {
                P->off_proc->vals[j] /= -weak_sum;
            }
        }
        pos[i] = -1;

        P->on_proc->idx1[i+1] = row_end_on;
        P->off_proc->idx1[i+1] = row_end_off;
    }
    P->on_proc->nnz = nnz_on;
    P->off_proc->nnz = nnz_off;

    P->on_proc->idx2.resize(nnz_on);
    P->on_proc->idx2.shrink_to_fit();
    P->on_proc->vals.resize(nnz_on);
    P->on_proc->vals.shrink_to_fit();

    P->off_proc->idx2.resize(nnz_off);
    P->off_proc->idx2.shrink_to_fit();
    P->off_proc->vals.resize(nnz_off);
    P->off_proc->vals.shrink_to_fit();

    P->local_nnz = P->on_proc->nnz + P->off_proc->nnz;

    // Update off_proc columns in P (remove col j if col_exists[j] is false)
    if (P->off_proc_num_cols)
    {
    	aligned_vector<int> P_to_new(P->off_proc_num_cols);
    	for (int i = 0; i < P->off_proc_num_cols; i++)
    	{
        	if (col_exists[i])
        	{
            	P_to_new[i] = P->off_proc_column_map.size();
            	P->off_proc_column_map.emplace_back(off_proc_column_map[i]);
        	}
    	}
        for (aligned_vector<int>::iterator it = P->off_proc->idx2.begin(); 
                it != P->off_proc->idx2.end(); ++it)
        {
            *it = P_to_new[*it];
        }
    }

    P->off_proc_num_cols = P->off_proc_column_map.size();
    P->on_proc_num_cols = P->on_proc_column_map.size();
    P->off_proc->n_cols = P->off_proc_num_cols;
    P->on_proc->n_cols = P->on_proc_num_cols;

    if (tap_interp)
    {
        P->init_tap_communicators(MPI_COMM_WORLD, comm_t);
    }
    else
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map,
                P->on_proc_column_map, 9243, MPI_COMM_WORLD, comm_t);
    }

    delete recv_mat;

    // Finish Allreduce and set global number of columns
    MPI_Wait(&reduce_request, MPI_STATUS_IGNORE);
    P->global_num_cols = global_num_cols;

    return P;
}

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states, 
        bool tap_interp, int num_variables, int* variables, 
        data_t* comm_t, data_t* comm_mat_t)
{
    int start, end;
    int start_k, end_k;
    int end_S;
    int col, col_k;
    int ctr, idx;
    int global_col, sign;
    int global_num_cols;
    int row_start_on, row_start_off;
    double diag, val, val_k;
    double weak_sum, coarse_sum;
    aligned_vector<int> off_variables;
    if (A->off_proc_num_cols) off_variables.resize(A->off_proc_num_cols);

    CommPkg* comm = A->comm;
    CommPkg* mat_comm = A->comm;
    if (tap_interp)
    {
        comm = A->tap_comm;
        mat_comm = A->tap_mat_comm;
    }

    CSRMatrix* recv_mat; // On Proc Block of Recvd A

    A->sort();
    S->sort();
    A->on_proc->move_diag();
    S->on_proc->move_diag();

    if (num_variables > 1)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        comm->communicate(variables);
        if (comm_t) *comm_t += MPI_Wtime();

        for (int i = 0; i < A->off_proc_num_cols; i++)
        {
            off_variables[i] = comm->get_int_buffer()[i];
        }
    }

    // Initialize P
    aligned_vector<int> on_proc_col_to_new;
    aligned_vector<int> off_proc_col_to_new;
    aligned_vector<bool> col_exists;
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
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == Selected)
        {
            on_proc_cols++;
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == Selected)
        {
            off_proc_cols++;
        }
    }
    MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
    ParCSRMatrix* P = new ParCSRMatrix(A->partition, A->global_num_rows, global_num_cols, 
            A->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (states[i] == Selected)
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.emplace_back(S->on_proc_column_map[i]);
        }
    }
    P->local_row_map = S->get_local_row_map();

    // Communicate parallel matrix A (Costly!)
    if (comm_mat_t) *comm_mat_t -= MPI_Wtime();
    recv_mat = communicate(A, states, off_proc_states, mat_comm);
    if (comm_mat_t) *comm_mat_t += MPI_Wtime();

    CSRMatrix* recv_on = new CSRMatrix(recv_mat->n_rows, -1, recv_mat->nnz);
    CSRMatrix* recv_off = new CSRMatrix(recv_mat->n_rows, -1, recv_mat->nnz);
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = recv_mat->idx2[j];
            if (col < A->partition->first_local_col || col > A->partition->last_local_col)
            {
                recv_off->idx2.emplace_back(col);
                recv_off->vals.emplace_back(recv_mat->vals[j]);
            }
            else
            {
                recv_on->idx2.emplace_back(col);
                recv_on->vals.emplace_back(recv_mat->vals[j]);
            }
        }
        recv_on->idx1[i+1] = recv_on->idx2.size();
        recv_off->idx1[i+1] = recv_off->idx2.size();
    }
    recv_on->nnz = recv_on->idx2.size();
    recv_off->nnz = recv_off->idx2.size();

    delete recv_mat;

    // Change on_proc_cols to local
    recv_on->n_cols = A->on_proc_num_cols;
    int* on_proc_partition_to_col = A->map_partition_to_local();
    for (aligned_vector<int>::iterator it = recv_on->idx2.begin();
            it != recv_on->idx2.end(); ++it)
    {
        *it = on_proc_partition_to_col[*it - A->partition->first_local_row];
    }
    delete[] on_proc_partition_to_col;

    // Change off_proc_cols to local (remove cols not on rank)
    ctr = 0;
    std::map<int, int> global_to_local;
    for (aligned_vector<int>::iterator it = A->off_proc_column_map.begin();
            it != A->off_proc_column_map.end(); ++it)
    {
        global_to_local[*it] = ctr++;
    }
    recv_off->n_cols = A->off_proc_num_cols;
    ctr = 0;
    start = recv_off->idx1[0];
    for (int i = 0; i < recv_off->n_rows; i++)
    {
        end = recv_off->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = recv_off->idx2[j];
            std::map<int, int>::iterator it = global_to_local.find(global_col);
            if (it != global_to_local.end())
            {
                recv_off->idx2[ctr] = it->second;
                recv_off->vals[ctr++] = recv_off->vals[j];
            }
        }
        recv_off->idx1[i+1] = ctr;
        start = end;
    }
    recv_off->nnz = ctr;
    recv_off->idx2.resize(ctr);
    recv_off->vals.resize(ctr);

    // For each row, will calculate coarse sums and store 
    // strong connections in vector
    aligned_vector<int> pos;
    aligned_vector<int> off_proc_pos;
    if (A->on_proc_num_cols)
    {
        pos.resize(A->on_proc_num_cols, -1);
    }
    if (A->off_proc_num_cols)
    {
        off_proc_pos.resize(A->off_proc_num_cols, -1);
    }

    P->on_proc->idx1[0] = 0;
    P->off_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        // If coarse row, add to P
        if (states[i] == Selected)
        {
            P->on_proc->idx2.emplace_back(on_proc_col_to_new[i]);
            P->on_proc->vals.emplace_back(1);
            P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
            P->off_proc->idx1[i+1] = P->off_proc->idx2.size();
            continue;
        }

        row_start_on = P->on_proc->idx1[i];
        row_start_off = P->off_proc->idx1[i];

        // Add selected states to P
        start = S->on_proc->idx1[i] + 1;
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            if (states[col] == Selected)
            {
                val = S->on_proc->vals[j];
                pos[col] = P->on_proc->idx2.size();
                P->on_proc->idx2.push_back(on_proc_col_to_new[col]);
                P->on_proc->vals.push_back(val);
            }
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];
            if (off_proc_states[col] == Selected)
            {
                val = S->off_proc->vals[j];
                off_proc_pos[col] = P->off_proc->idx2.size();
                col_exists[col] = true;
                P->off_proc->idx2.push_back(col);
                P->off_proc->vals.push_back(val);
            }
        }

        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        ctr = S->on_proc->idx1[i]+1;
        end_S = S->on_proc->idx1[i+1];
        weak_sum = A->on_proc->vals[start++];
        for (int j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            if (ctr < end_S && S->on_proc->idx2[ctr] == col)
            {
                ctr++;

                if (states[col] == Selected) continue;

                // Find sum of all coarse points in row k (with sign NOT equal to diag)
                coarse_sum = 0;
                start_k = A->on_proc->idx1[col];
                end_k = A->on_proc->idx1[col+1];

                diag = A->on_proc->vals[start_k++];
                if (diag > 0) sign = 1;
                else sign = -1;

                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->on_proc->idx2[k];
                    if (states[col_k] == Selected)
                    {
                        val_k = A->on_proc->vals[k];
                        if (val_k * sign < 0 && pos[col_k] >= row_start_on)
                        {
                            coarse_sum += val_k;
                        }
                    }
                }

                start_k = A->off_proc->idx1[col];
                end_k = A->off_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->off_proc->idx2[k];
                    val_k = A->off_proc->vals[k];
                    if (val_k * sign < 0 && off_proc_pos[col_k] >= row_start_off)
                    {
                        coarse_sum += val_k;
                    }
                }
                        
                if (fabs(coarse_sum) < zero_tol)
                {
                    weak_sum += val;
                }
                else
                {
                    coarse_sum = val / coarse_sum;
                }

                if (coarse_sum) // k in D_i^S
                {
                    start_k = A->on_proc->idx1[col]+1;
                    end_k = A->on_proc->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = A->on_proc->idx2[k];
                        if (states[col_k] == Selected)
                        {
                            val_k = A->on_proc->vals[k];
                            idx = pos[col_k];
                            if (val_k * sign < 0 && idx >= row_start_on)
                            {
                                P->on_proc->vals[idx] += (coarse_sum * val_k);
                            }
                        }
                    }

                    start_k = A->off_proc->idx1[col];
                    end_k = A->off_proc->idx1[col+1];
                    for (int k = start_k; k < end_k; k++)
                    {
                        col_k = A->off_proc->idx2[k];
                        if (off_proc_states[col_k] == Selected)
                        {
                            val_k = A->off_proc->vals[k];
                            idx = off_proc_pos[col_k];
                            if (val_k * sign < 0 && idx >= row_start_off)
                            {
                                P->off_proc->vals[idx] += (coarse_sum * val_k);
                            }
                        }
                    }
                }
            }
            else if (num_variables == 1 || variables[i] == variables[col])
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
            if (ctr < end_S && S->off_proc->idx2[ctr] == col)
            {
                ctr++;

                if (off_proc_states[col] == Selected) continue;

                // Strong connection... create 
                coarse_sum = 0;
                start_k = recv_on->idx1[col];
                end_k = recv_on->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = recv_on->idx2[k];
                    val_k = recv_on->vals[k];
                    if (pos[col_k] >= row_start_on)
                    {
                        coarse_sum += val_k;
                    }
                }
                start_k = recv_off->idx1[col];
                end_k = recv_off->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = recv_off->idx2[k];
                    val_k = recv_off->vals[k];
                    if (off_proc_pos[col_k] >= row_start_off)
                    {
                        coarse_sum += val_k;
                    }
                }
                if (fabs(coarse_sum) < zero_tol)
                {
                    weak_sum += val;
                }
                else
                {
                    coarse_sum = val / coarse_sum;
                }

                start_k = recv_on->idx1[col];
                end_k = recv_on->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    val_k = recv_on->vals[k];
                    col_k = recv_on->idx2[k];
                    idx = pos[col_k];
                    if (idx >= row_start_on)
                    {
                        P->on_proc->vals[idx] += (coarse_sum * val_k);
                    }
                }

                start_k = recv_off->idx1[col];
                end_k = recv_off->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    val_k = recv_off->vals[k];
                    col_k = recv_off->idx2[k];
                    idx = off_proc_pos[col_k];
                    if (idx >= row_start_off)
                    {
                        P->off_proc->vals[idx] += (coarse_sum * val_k);
                    }
                }
            }
            else if (num_variables == 1 || variables[i] == off_variables[col])
            {
                weak_sum += val;
            }
        }

        P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
        P->off_proc->idx1[i+1] = P->off_proc->idx2.size();

        start = P->on_proc->idx1[i];
        end = P->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            P->on_proc->vals[j] /= -weak_sum;
        }
        start = P->off_proc->idx1[i];
        end = P->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            P->off_proc->vals[j] /= -weak_sum;

        }
    }
    P->on_proc->nnz = P->on_proc->idx2.size();
    P->off_proc->nnz = P->off_proc->idx2.size();
    P->local_nnz = P->on_proc->nnz + P->off_proc->nnz;

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            off_proc_col_to_new[i] = P->off_proc_column_map.size();
            P->off_proc_column_map.emplace_back(S->off_proc_column_map[i]);
        }
    }
    for (aligned_vector<int>::iterator it = P->off_proc->idx2.begin(); 
            it != P->off_proc->idx2.end(); ++it)
    {
        *it = off_proc_col_to_new[*it];
    }

    P->off_proc_num_cols = P->off_proc_column_map.size();
    P->on_proc_num_cols = P->on_proc_column_map.size();
    P->off_proc->n_cols = P->off_proc_num_cols;
    P->on_proc->n_cols = P->on_proc_num_cols;

    if (tap_interp)
    {
        P->update_tap_comm(S, on_proc_col_to_new, off_proc_col_to_new, comm_t);
    }
    else
    {
        P->comm = new ParComm(S->comm, on_proc_col_to_new, off_proc_col_to_new,
                comm_t);
    }

    delete recv_on;
    delete recv_off;

    return P;
}


ParCSRMatrix* direct_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states, 
        bool tap_interp, data_t* comm_t)
{
    int start, end, col;
    int global_num_cols;
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
    aligned_vector<double> sa_on;
    aligned_vector<double> sa_off;
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

    aligned_vector<int> on_proc_col_to_new;
    aligned_vector<int> off_proc_col_to_new;
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
    MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
    ParCSRMatrix* P = new ParCSRMatrix(S->partition, S->global_num_rows, global_num_cols, 
            S->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i])
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.emplace_back(S->on_proc_column_map[i]);
        }
    }
    aligned_vector<bool> col_exists;
    if (S->off_proc_num_cols)
    {
        col_exists.resize(S->off_proc_num_cols, false);
    }
    P->local_row_map = S->get_local_row_map();

    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] == 1)
        {
            P->on_proc->idx2.emplace_back(on_proc_col_to_new[i]);
            P->on_proc->vals.emplace_back(1);
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
                    P->on_proc->idx2.emplace_back(on_proc_col_to_new[col]);
                    
                    if (val < 0)
                    {
                        P->on_proc->vals.emplace_back(neg_coeff * val);
                    }
                    else
                    {
                        P->on_proc->vals.emplace_back(pos_coeff * val);
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
                    P->off_proc->idx2.emplace_back(col);

                    if (val < 0)
                    {
                        P->off_proc->vals.emplace_back(neg_coeff * val);
                    }
                    else
                    {
                        P->off_proc->vals.emplace_back(pos_coeff * val);
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
            P->off_proc_column_map.emplace_back(S->off_proc_column_map[i]);
        }
    }
    for (aligned_vector<int>::iterator it = P->off_proc->idx2.begin(); 
            it != P->off_proc->idx2.end(); ++it)
    {
        *it = off_proc_col_to_new[*it];
    }


    P->off_proc_num_cols = P->off_proc_column_map.size();
    P->on_proc_num_cols = P->on_proc_column_map.size();
    P->off_proc->n_cols = P->off_proc_num_cols;
    P->on_proc->n_cols = P->on_proc_num_cols;

    if (tap_interp)
    {
        P->update_tap_comm(S, on_proc_col_to_new, off_proc_col_to_new, comm_t);
    }
    else
    {
        P->comm = new ParComm(S->comm, on_proc_col_to_new, off_proc_col_to_new,
                comm_t);
    }


    return P;
}
