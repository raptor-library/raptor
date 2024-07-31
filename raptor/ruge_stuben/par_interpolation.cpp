// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/ruge_stuben/par_interpolation.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <optional>

namespace raptor {

// Declare Private Methods
CSRMatrix* communicate(ParCSRMatrix* A, ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, CommPkg* comm);
CSRMatrix*  communicate(ParCSRMatrix* A, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, CommPkg* comm);
void filter_interp(ParCSRMatrix* P, const double filter_threshold);
ParCSRMatrix* extended_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, const double filter_threshold,
        bool tap_interp, int num_variables, int* variables);
ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        bool tap_interp, int num_variables, int* variables);
ParCSRMatrix* direct_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, bool tap_interp);



// TODO -- if in S, col is positive, otherwise col is -(col+1)
CSRMatrix* communicate(ParCSRMatrix* A, ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, CommPkg* comm)
{
    int start, end, col;
    int ctr_S, end_S, global_col;
    int sign;
    double diag, val;

	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> rowptr(A->local_num_rows + 1);
    std::vector<int> col_indices;
    std::vector<double> values;
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
                        col_indices.push_back(-(global_col+1));
                        values.push_back(val);
                    }
                    ctr_S++;
                }
                else
                {
                    if (val * sign < 0) // Only add needed cols
                    {
                        col_indices.push_back(global_col);
                        values.push_back(val);
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
                    col_indices.push_back(global_col);
                    values.push_back(val);
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
                    col_indices.push_back(-(global_col+1));
                    values.push_back(val);
                }
                ctr_S++;
            }
            else
            {
                // Selected weak connection
                if (val * sign < 0)
                {
                    col_indices.push_back(global_col);
                    values.push_back(val);
                }
            }
        }
        rowptr[i+1] = col_indices.size();
    }

    return comm->communicate(rowptr, col_indices, values);

}


CSRMatrix*  communicate(ParCSRMatrix* A, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, CommPkg* comm)
{
    int start, end, col;
    int sign;
    double val;

    std::vector<int> rowptr(A->local_num_rows + 1);
    std::vector<int> col_indices;
    std::vector<double> values;
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
                col_indices.push_back(A->on_proc_column_map[col]);
                values.push_back(A->on_proc->vals[j]);
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
                col_indices.push_back(A->off_proc_column_map[col]);
                values.push_back(A->off_proc->vals[j]);
            }
        }
        rowptr[i+1] = col_indices.size();
    }

    return comm->communicate(rowptr, col_indices, values);
}

void filter_interp(ParCSRMatrix* P, const double filter_threshold)
{
    int row_start_on = 0;
    int row_start_off = 0;
    int row_end_on, row_end_off;
    int ctr_on = 0;
    int ctr_off = 0;
    int prev_ctr_on, prev_ctr_off;

    double val, abs_val;
    double row_max, row_sum, row_scale;
    double remain_sum;

    if (filter_threshold > zero_tol && filter_threshold <= 1)
    {
        for (int i = 0; i < P->local_num_rows; i++)
        {
            prev_ctr_on = ctr_on;
            prev_ctr_off = ctr_off;

            row_end_on = P->on_proc->idx1[i+1];
            row_end_off = P->off_proc->idx1[i+1];

            row_max = 0;
            for (int j = row_start_on; j < row_end_on; j++)
            {
                val = P->on_proc->vals[j];
                abs_val = fabs(val);
                if (abs_val > row_max)
                    row_max = abs_val;
            }
            for (int j = row_start_off; j < row_end_off; j++)
            {
                val = P->off_proc->vals[j];
                abs_val = fabs(val);
                if (abs_val > row_max)
                    row_max = abs_val;
            }

            row_max *= filter_threshold;
            row_sum = 0;
            remain_sum = 0;
            for (int j = row_start_on; j < row_end_on; j++)
            {
                val = P->on_proc->vals[j];
                row_sum += val;
                if (fabs(val) >= row_max)
                {
                    P->on_proc->idx2[ctr_on] = P->on_proc->idx2[j];
                    P->on_proc->vals[ctr_on] = val;
                    ctr_on++;
                    remain_sum += val;
                }
            }
            for (int j = row_start_off; j < row_end_off; j++)
            {
                val = P->off_proc->vals[j];
                row_sum += val;
                if (fabs(val) >= row_max)
                {
                    P->off_proc->idx2[ctr_off] = P->off_proc->idx2[j];
                    P->off_proc->vals[ctr_off] = val;
                    ctr_off++;
                    remain_sum += val;
                }
            }

            if (fabs(remain_sum) > zero_tol && fabs(row_sum - remain_sum) > zero_tol)
            {
                row_scale = row_sum / remain_sum;
                for (int j = prev_ctr_on; j < ctr_on; j++)
                    P->on_proc->vals[j] *= row_scale;
                for (int j = prev_ctr_off; j < ctr_off; j++)
                    P->off_proc->vals[j] *= row_scale;
            }

            P->on_proc->idx1[i+1] = ctr_on;
            P->off_proc->idx1[i+1] = ctr_off;

            row_start_on = row_end_on;
            row_start_off = row_end_off;
        }
    }
    else
    {
        ctr_on = P->on_proc->idx2.size();
        ctr_off = P->off_proc->idx2.size();
    }

    P->on_proc->nnz = ctr_on;
    P->off_proc->nnz = ctr_off;
    P->local_nnz = ctr_on + ctr_off;

    P->on_proc->idx2.resize(ctr_on);
    P->on_proc->vals.resize(ctr_on);
    P->off_proc->idx2.resize(ctr_off);
    P->off_proc->vals.resize(ctr_off);

    P->on_proc->idx2.shrink_to_fit();
    P->on_proc->vals.shrink_to_fit();
    P->off_proc->idx2.shrink_to_fit();
    P->off_proc->vals.shrink_to_fit();
}


ParCSRMatrix* extended_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        const double filter_threshold,
        bool tap_interp, int num_variables, int* variables)
{
    int start, end, idx;
    int ctr, end_S;
    int start_k, end_k;
    int col;
    int col_k, col_P;
    int global_num_cols;
    int on_proc_cols, off_proc_cols;
    int sign;
    int row_start_on, row_start_off;
    int row_end_on = 0;
    int row_end_off = 0;
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
    std::vector<int> off_proc_column_map;
    std::vector<int> off_variables;

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
        comm->communicate(variables);

        for (int i = 0; i < A->off_proc_num_cols; i++)
        {
            off_variables[i] = comm->get_int_buffer()[i];
        }
    }

    // Communicate parallel matrix A (portion needed)
    recv_mat = communicate(A, S, states, off_proc_states, mat_comm);

    int tmp_col;
    auto on_proc_partition_to_col = A->map_partition_to_local();

    std::vector<int> A_recv_on_ptr(recv_mat->n_rows + 1);
    std::vector<int> S_recv_on_ptr(recv_mat->n_rows + 1);
    std::vector<int> A_recv_off_ptr(recv_mat->n_rows + 1);
    std::vector<int> S_recv_off_ptr(recv_mat->n_rows + 1);
    std::vector<int> A_recv_on_idx(recv_mat->nnz);
    std::vector<int> S_recv_on_idx(recv_mat->nnz);
    std::vector<int> A_recv_off_idx(recv_mat->nnz);
    std::vector<int> S_recv_off_idx(recv_mat->nnz);

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
        off_proc_column_map.push_back(*it);
    }
    off_proc_cols = off_proc_column_map.size();

    for (std::vector<int>::iterator it = A_recv_off_idx.begin();
            it != A_recv_off_idx.end(); ++it)
    {
        recv_mat->idx2[*it] = global_to_local[recv_mat->idx2[*it]];
    }

    // Initialize P
    std::vector<int> on_proc_col_to_new;
    std::vector<bool> col_exists;
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
    RAPtor_MPI_Request reduce_request;
    int reduce_buf = on_proc_cols;
    RAPtor_MPI_Iallreduce(&(reduce_buf), &global_num_cols, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM,
            RAPtor_MPI_COMM_WORLD, &reduce_request);

    ParCSRMatrix* P = new ParCSRMatrix(A->partition, A->global_num_rows, -1,
            A->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == Selected)
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.push_back(S->on_proc_column_map[i]);
        }
    }
    P->local_row_map = S->get_local_row_map();

    std::vector<int> off_proc_A_to_P;
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
    std::vector<int> pos;
    std::vector<int> off_proc_pos;
    std::vector<double> coarse_sum;
    std::vector<double> off_proc_coarse_sum;
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
            col_P = off_proc_A_to_P[col];
            idx = -1;
            if (col_P > -1) idx = off_proc_pos[col_P];
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
                        weak_sum += val;
                    }
                }
            }
        }

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

    filter_interp(P, filter_threshold);

    // Update off_proc columns in P (remove col j if col_exists[j] is false)
    if (P->off_proc_num_cols)
    {
    	std::vector<int> P_to_new(P->off_proc_num_cols);
    	for (int i = 0; i < P->off_proc_num_cols; i++)
    	{
        	if (col_exists[i])
        	{
            	P_to_new[i] = P->off_proc_column_map.size();
            	P->off_proc_column_map.push_back(off_proc_column_map[i]);
        	}
    	}
        for (std::vector<int>::iterator it = P->off_proc->idx2.begin();
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
        P->init_tap_communicators(RAPtor_MPI_COMM_WORLD);
    }
    else
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map,
                P->on_proc_column_map, 9243, RAPtor_MPI_COMM_WORLD);
    }

    delete recv_mat;

    // Finish Allreduce and set global number of columns
    RAPtor_MPI_Wait(&reduce_request, RAPtor_MPI_STATUS_IGNORE);
    P->global_num_cols = global_num_cols;

    return P;
}

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        bool tap_interp, int num_variables, int* variables)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

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
    std::vector<int> off_variables;
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
        comm->communicate(variables);

        for (int i = 0; i < A->off_proc_num_cols; i++)
        {
            off_variables[i] = comm->get_int_buffer()[i];
        }
    }

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
    RAPtor_MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);

    ParCSRMatrix* P = new ParCSRMatrix(A->partition, A->global_num_rows, global_num_cols,
            A->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (states[i] == Selected)
        {
            on_proc_col_to_new[i] = P->on_proc_column_map.size();
            P->on_proc_column_map.push_back(S->on_proc_column_map[i]);
        }
    }
    P->local_row_map = S->get_local_row_map();

    // Communicate parallel matrix A (Costly!)
    recv_mat = communicate(A, states, off_proc_states, mat_comm);

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
                recv_off->idx2.push_back(col);
                recv_off->vals.push_back(recv_mat->vals[j]);
            }
            else
            {
                recv_on->idx2.push_back(col);
                recv_on->vals.push_back(recv_mat->vals[j]);
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
    auto on_proc_partition_to_col = A->map_partition_to_local();
    for (std::vector<int>::iterator it = recv_on->idx2.begin();
            it != recv_on->idx2.end(); ++it)
    {
        *it = on_proc_partition_to_col[*it - A->partition->first_local_row];
    }

    // Change off_proc_cols to local (remove cols not on rank)
    ctr = 0;
    std::map<int, int> global_to_local;
    for (std::vector<int>::iterator it = A->off_proc_column_map.begin();
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
    std::vector<int> pos;
    std::vector<int> off_proc_pos;
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
            P->on_proc->idx2.push_back(on_proc_col_to_new[i]);
            P->on_proc->vals.push_back(1);
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
            else if (states[col] != NoNeighbors)
            {
                if (num_variables == 1 || variables[i] == variables[col])
                {
                    weak_sum += val;
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
            else if (off_proc_states[col] != NoNeighbors)
            {
                if (num_variables == 1 || variables[i] == off_variables[col])
                {
                    weak_sum += val;
                }
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

    if (tap_interp)
    {
        P->update_tap_comm(S, on_proc_col_to_new, off_proc_col_to_new);
    }
    else
    {
        P->comm = new ParComm(S->comm, on_proc_col_to_new, off_proc_col_to_new);
    }

    delete recv_on;
    delete recv_off;

    return P;
}


ParCSRMatrix* direct_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        bool tap_interp)
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
    RAPtor_MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);

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

    if (tap_interp)
    {
        P->update_tap_comm(S, on_proc_col_to_new, off_proc_col_to_new);
    }
    else
    {
        P->comm = new ParComm(S->comm, on_proc_col_to_new, off_proc_col_to_new);
    }


    return P;
}

namespace one_point {
namespace {
ParCSRMatrix * create_P(const ParCSRMatrix & A,
                        const ParCSRMatrix & S,
                        const splitting_t splitting) {
	bool isbsr = dynamic_cast<const ParBSRMatrix*>(&A);

	auto count_cols = [](int n, const std::vector<int> & split) {
		int cols{0};
		for (int i = 0; i < n; ++i) {
			if (split[i]) ++cols;
		}
		return cols;
	};
	auto on_proc_cols = count_cols(S.on_proc_num_cols, splitting.on_proc);
	auto off_proc_cols = count_cols(S.off_proc_num_cols, splitting.off_proc);
	auto global_cols = [](int local) {
		int global;
		RAPtor_MPI_Allreduce(&local, &global, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
		return global;
	}(on_proc_cols);

	if (isbsr) {
		auto & bsr = dynamic_cast<const ParBSRMatrix&>(A);
		return new ParBSRMatrix(S.partition, S.global_num_rows, global_cols,
		                        S.local_num_rows, on_proc_cols, off_proc_cols,
		                        bsr.on_proc->b_rows, bsr.on_proc->b_cols);
	} else {
		return new ParCSRMatrix(S.partition, S.global_num_rows, global_cols,
		                        S.local_num_rows, on_proc_cols, off_proc_cols);
	}
}

std::vector<int> map_on_proc_cpoints(ParCSRMatrix & P,
                                     const std::vector<int> & colmap,
                                     const decltype(splitting_t::on_proc) & splitting) {
	std::vector<int> cpoint_map;
	cpoint_map.resize(colmap.size(), -1);

	for (std::size_t i = 0; i < colmap.size(); ++i) {
		if (splitting[i] == Selected) {
			cpoint_map[i] = P.on_proc_column_map.size();
			P.on_proc_column_map.push_back(colmap[i]);
		}
	}

	return cpoint_map;
}

void push_identity(Matrix & mat) {
	bool isbsr = dynamic_cast<BSRMatrix*>(&mat);
	if (isbsr) {
		auto & bsr = dynamic_cast<BSRMatrix&>(mat);
		auto bvals = new double[bsr.b_size]();
		for (int i = 0; i < bsr.b_rows; ++i) {
			bvals[i * bsr.b_cols + i] = 1.;
		}
		bsr.block_vals.push_back(bvals);
	} else {
		mat.vals.push_back(1.);
	}
}

inline bool find_strongest(int row,
                           const Matrix & soc_mat,
                           const decltype(splitting_t::on_proc) & splitting,
                           double & max,
                           int & ind) {
	auto & soc = dynamic_cast<const CSRMatrix&>(soc_mat);
	bool found = false;
	for (int off = soc.idx1[row]; off < soc.idx1[row+1]; ++off) {
		auto col = soc.idx2[off];
		if (splitting[col] == Selected) {
			double vv = std::abs(soc.vals[off]);
			if (vv > max) {
				max = vv;
				ind = col;
				found = true;
			}
		}
	}
	return found;
}

void set_nnz(ParCSRMatrix & P) {
	auto set = [](Matrix & mat) {
		mat.nnz = mat.idx2.size();
	};
	set(*P.on_proc);
	set(*P.off_proc);
	P.local_nnz = P.on_proc->nnz + P.off_proc->nnz;
}

std::vector<int> remap_off_proc(ParCSRMatrix & P,
                                const ParCSRMatrix & S,
                                const std::vector<bool> & col_exists) {
	std::vector<int> cpoint_map;
	if (S.off_proc_num_cols) {
		cpoint_map.resize(S.off_proc_num_cols, -1);
	}

	for (int i = 0; i < S.off_proc_num_cols; ++i) {
		if (col_exists[i]) {
			cpoint_map[i] = P.off_proc_column_map.size();
			P.off_proc_column_map.push_back(S.off_proc_column_map[i]);
		}
	}

	std::transform(
		P.off_proc->idx2.cbegin(),
		P.off_proc->idx2.cend(),
		P.off_proc->idx2.begin(),
		[&](int ind) { return cpoint_map[ind]; });

	return cpoint_map;
}


void finalize_sizes(ParCSRMatrix & P) {
	P.off_proc_num_cols = P.off_proc_column_map.size();
	P.on_proc_num_cols = P.on_proc_column_map.size();
	P.off_proc->n_cols = P.off_proc_num_cols;
	P.on_proc->n_cols = P.on_proc_num_cols;
}

} // namespace
} // namespace one_point

ParCSRMatrix *one_point_interpolation(const ParCSRMatrix & A,
                                      const ParCSRMatrix & S,
                                      const splitting_t & splitting) {
	using namespace one_point;
	auto *P = create_P(A, S, splitting);
	// map cpoints in S and splitting to P indexing
	auto on_proc_cpoint_map = map_on_proc_cpoints(*P, S.on_proc_column_map, splitting.on_proc);
	P->get_local_row_map() = S.get_local_row_map();

	std::vector<bool> col_exists;
	if (S.off_proc_num_cols) col_exists.resize(S.off_proc_num_cols, false);

	for (int i = 0; i < A.local_num_rows; ++i) {
		if (splitting.on_proc[i] == Selected) {
			// Set C-point as identity
			P->on_proc->idx2.push_back(on_proc_cpoint_map[i]);
			push_identity(*P->on_proc);
		} else {
			/* find strongest connection to C-point and interpolate
			   directly from the C-point.
			*/
			double max = -1;
			int ind = -1;
			// search on_proc
			find_strongest(i, *S.on_proc,
			               splitting.on_proc,
			               max, ind);
			// search off_proc
			bool found_off_proc =
				find_strongest(i, *S.off_proc,
				               splitting.off_proc,
				               max, ind);
			if (found_off_proc) {
				col_exists[ind] = true;
				P->off_proc->idx2.push_back(ind);
				push_identity(*P->off_proc);
			} else if (ind > -1) {
				P->on_proc->idx2.push_back(on_proc_cpoint_map[ind]);
				push_identity(*P->on_proc);
			}
		}
		auto inc_rowptr = [=](Matrix & mat) {
			mat.idx1[i+1] = mat.idx2.size();
		};
		inc_rowptr(*P->on_proc);
		inc_rowptr(*P->off_proc);
	}
	set_nnz(*P);

	// remap off_proc columns
	auto off_proc_cpoint_map = remap_off_proc(*P, S, col_exists);

	finalize_sizes(*P);

	P->comm = new ParComm(S.comm, on_proc_cpoint_map, off_proc_cpoint_map);

	return P;
}

ParBSRMatrix * one_point_interpolation(const ParBSRMatrix & A,
                                       const ParCSRMatrix & S,
                                       const splitting_t & splitting)
{
	auto ret = dynamic_cast<ParBSRMatrix*>(
		one_point_interpolation(
			dynamic_cast<const ParCSRMatrix&>(A), S, splitting));

	assert(ret);
	return ret;
}

namespace lair {
namespace {


/*
  Helper type providing access to received rows based
  on whether they are on_proc or off_proc
*/
struct comm_rows {
	comm_rows(const ParCSRMatrix & A,
	          CSRMatrix * mat);
	~comm_rows() {
		if (rmat) delete rmat;
	}

	template<class F>
	void iter_diag(int row, F && f) const {
		iter(row, std::forward<F>(f), diag);
	}
	template<class F>
	void iter_offd(int row, F && f) const {
		iter(row, std::forward<F>(f), offd);
	}

	struct ptrs {
		std::vector<int> ptr;
		std::vector<int> idx;
	} diag, offd;

	template<class T>
	struct row_view {
		auto & idx2() {	return mat->idx2[off]; }
		auto & val() { return mat->vals[off]; };

		int off;
		T * mat;
	};

private:
	template<class F>
	void iter(int row, F && f, const ptrs & pts) const {
		for (int i = pts.ptr[row]; i < pts.ptr[row + 1]; ++i) {
			std::forward<F>(f)(row_view<const CSRMatrix>{i, rmat});
		}
	}

	CSRMatrix *rmat;
};

struct offd_map_rowptr {
	std::vector<int> diag_rowptr;
	std::vector<int> offd_rowptr;
	std::vector<int> offd_colmap; //offd colmap
};

/*
  First pass: compute rowptr for R and discover off proc columns.

  This computes the rowptr for R and discovers set of off proc columns.
  Forward and backward column maps for off proc columns are also computed.
 */
offd_map_rowptr map_offd_fill_rowptr(const ParCSRMatrix & S,
                                     const splitting_t & splitting,
                                     const std::vector<int> & cpts,
                                     fpoint_distance distance,
                                     const comm_rows & recv_rows) {
	offd_map_rowptr ret;

	struct nnz_t {
		int diag = 0;
		int offd = 0;
	} nnz;

	[&](auto & ... v) {
		((v.resize(cpts.size() + 1)), ...);
		((v[0] = 0), ...);
	}(ret.diag_rowptr, ret.offd_rowptr);

	std::set<int> offd_cols;
	auto expand = [&](const int row,
	                  const auto & soc,
	                  const auto & split,
	                  auto && callback) {
		for (int off = soc.idx1[row]; off < soc.idx1[row+1]; ++off) {
			auto d2point = soc.idx2[off];
			if (split[d2point] == Unselected &&
			    d2point != row){
				std::forward<decltype(callback)>(callback)(d2point);
			}
		}
	};
	for (std::size_t row = 0; row < cpts.size(); ++row) {
		auto cpoint = cpts[row];
		for (int off = S.on_proc->idx1[cpoint];
		     off < S.on_proc->idx1[cpoint + 1]; ++off) {
			auto this_point = S.on_proc->idx2[off];
			if (splitting.on_proc[this_point] == Unselected) {
				++nnz.diag;
				if (distance == fpoint_distance::two) {
					expand(this_point, *S.on_proc, splitting.on_proc, [&](int){ ++nnz.diag; });
					expand(this_point, *S.off_proc, splitting.off_proc,
					       [&](int col) {
						       ++nnz.offd;
						       offd_cols.insert(S.off_proc_column_map[col]);
					       });
				}
			}
		}
		for (int off = S.off_proc->idx1[cpoint];
		     off < S.off_proc->idx1[cpoint + 1]; ++off) {
			auto this_point = S.off_proc->idx2[off];
			if (splitting.off_proc[this_point] == Unselected) {
				++nnz.offd;
				auto global_col = S.off_proc_column_map[this_point];
				offd_cols.insert(global_col);
				if (distance == fpoint_distance::two) {
					recv_rows.iter_diag(this_point, [&](auto) {
						++nnz.diag;
					});
					recv_rows.iter_offd(this_point, [&](auto rview) {
						if (rview.idx2() != global_col) {
							++nnz.offd;
							offd_cols.insert(rview.idx2());
						}
					});
				}
			}
		}
		++nnz.diag; // identity on cpoints
		ret.diag_rowptr[row + 1] = nnz.diag;
		ret.offd_rowptr[row + 1] = nnz.offd;
	}

	for (auto col : offd_cols) {
		ret.offd_colmap.push_back(col);
	}

	return ret;
}
ParCSRMatrix * create_R(const ParCSRMatrix & A,
                        const ParCSRMatrix & S,
                        const splitting_t & splitting,
                        offd_map_rowptr && rowptr_and_colmap) {


	bool isbsr = dynamic_cast<const ParBSRMatrix*>(&A);

	int local_rows = std::count(splitting.on_proc.cbegin(),
	                            splitting.on_proc.cend(),
	                            Selected);
	auto global_rows = [](int local) {
		int global;
		RAPtor_MPI_Allreduce(&local, &global, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
		return global;
	}(local_rows);

	auto off_proc_num_cols = rowptr_and_colmap.offd_colmap.size();
	auto move_data = [&](offd_map_rowptr && rac, ParMatrix & mat) {
		mat.on_proc->idx1 = std::move(rac.diag_rowptr);
		mat.off_proc->idx1 = std::move(rac.offd_rowptr);
		mat.off_proc_column_map = std::move(rac.offd_colmap);
		mat.on_proc_column_map = A.on_proc_column_map;
		[](auto & ... mats) {
			(mats.idx2.resize(mats.idx1.back()), ...);
			(mats.vals.resize(mats.idx1.back()), ...);
		}(*mat.on_proc, *mat.off_proc);
	};
	if (isbsr) {
		auto & bsr = dynamic_cast<const ParBSRMatrix&>(A);
		auto * R = new ParBSRMatrix(S.partition, global_rows, S.global_num_cols,
		                            local_rows, S.on_proc_num_cols, off_proc_num_cols,
		                            bsr.on_proc->b_rows, bsr.on_proc->b_cols);
		move_data(std::move(rowptr_and_colmap), *R);
		return R;
	} else {
		auto * R = new ParCSRMatrix(S.partition, global_rows, S.global_num_cols,
		                            local_rows, S.on_proc_num_cols, off_proc_num_cols);
		move_data(std::move(rowptr_and_colmap), *R);
		return R;
	}
}


ParComm create_neighborhood_comm(const ParCSRMatrix & R, const ParCSRMatrix & A) {
	constexpr int tag = 9345;
	return ParComm(A.partition, R.off_proc_column_map,
	               A.on_proc_column_map, tag, RAPtor_MPI_COMM_WORLD);
}


auto get_cpoints(const std::vector<int> & split) {
	std::vector<int> cpts;

	for (std::size_t i = 0; i < split.size(); ++i) {
		if (split[i] == Selected)
			cpts.push_back(i);
	}

	return cpts;
}

namespace pyamg {
/* Sign-of Function overloaded for int, float and double
 * signof(x) =  1 if x > 0
 * signof(x) = -1 if x < 0
 * signof(0) =  1 if x = 0
 */
inline int signof(int a) { return (a<0 ? -1 : 1); }
inline float signof(float a) { return (a<0.0 ? -1.0 : 1.0); }
inline double signof(double a) { return (a<0.0 ? -1.0 : 1.0); }

/*
 * Return row-major index from 2d array index, A[row,col].
 */
template<class I>
inline I row_major(const I row, const I col, const I num_cols)
{
    return row*num_cols + col;
}

/*
 * Return column-major index from 2d array index, A[row,col].
 */
template<class I>
inline I col_major(const I row, const I col, const I num_rows)
{
    return col*num_rows + row;
}

/* QR-decomposition using Householer transformations on dense
 * 2d array stored in either column- or row-major form.
 *
 * Parameters
 * ----------
 * A : double array
 *     2d matrix A stored in 1d column- or row-major.
 * m : &int
 *     Number of rows in A
 * n : &int
 *     Number of columns in A
 * is_col_major : bool
 *     True if A is stored in column-major, false
 *     if A is stored in row-major.
 *
 * Returns
 * -------
 * Q : vector<double>
 *     Matrix Q stored in same format as A.
 * R : in-place
 *     R is stored over A in place, in same format.
 *
 * Notes
 * ------
 * Currently only set up for real-valued matrices. May easily
 * generalize to complex, but haven't checked.
 *
 */
template<class I, class T>
std::vector<T> QR(T A[],
                  const I &m,
                  const I &n,
                  const I is_col_major)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Initialize Q to identity
    std::vector<T> Q(m*m,0);
    for (I i=0; i<m; i++) {
        Q[get_ind(i,i,m)] = 1;
    }

    // Loop over columns of A using Householder reflections
    for (I j=0; j<n; j++) {

        // Break loop for short fat matrices
        if (m <= j) {
            break;
        }

        // Get norm of next column of A to be reflected. Choose sign
        // opposite that of A_jj to avoid catastrophic cancellation.
        // Skip loop if norm is zero, as that means column of A is all
        // zero.
        T normx = 0;
        for (I i=j; i<m; i++) {
            T temp = A[get_ind(i,j,*C)];
            normx += temp*temp;
        }
        normx = std::sqrt(normx);
        if (normx < 1e-12) {
            continue;
        }
        normx *= -1*signof(A[get_ind(j,j,*C)]);

        // Form vector v for Householder matrix H = I - tau*vv^T
        // where v = R(j:end,j) / scale, v[0] = 1.
        T scale = A[get_ind(j,j,*C)] - normx;
        T tau = -scale / normx;
        std::vector<T> v(m-j,0);
        v[0] = 1;
        for (I i=1; i<(m-j); i++) {
            v[i] = A[get_ind(j+i,j,*C)] / scale;
        }

        // Modify R in place, R := H*R, looping over columns then rows
        for (I k=j; k<n; k++) {

            // Compute the kth element of v^T * R
            T vtR_k = 0;
            for (I i=0; i<(m-j); i++) {
                vtR_k += v[i] * A[get_ind(j+i,k,*C)];
            }

            // Correction for each row of kth column, given by
            // R_ik -= tau * v_i * (vtR_k)_k
            for (I i=0; i<(m-j); i++) {
                A[get_ind(j+i,k,*C)] -= tau * v[i] * vtR_k;
            }
        }

        // Modify Q in place, Q = Q*H
        for (I i=0; i<m; i++) {

            // Compute the ith element of Q * v
            T Qv_i = 0;
            for (I k=0; k<(m-j); k++) {
                Qv_i += v[k] * Q[get_ind(i,k+j,m)];
            }

            // Correction for each column of ith row, given by
            // Q_ik -= tau * Qv_i * v_k
            for (I k=0; k<(m-j); k++) {
                Q[get_ind(i,k+j,m)] -= tau * v[k] * Qv_i;
            }
        }
    }

    return Q;
}

/* Backward substitution solve on upper-triangular linear system,
 * Rx = rhs, where R is stored in column- or row-major form.
 *
 * Parameters
 * ----------
 * R : double array, length m*n
 *     Upper-triangular array stored in column- or row-major.
 * rhs : double array, length m
 *     Right hand side of linear system
 * x : double array, length n
 *     Preallocated array for solution
 * m : &int
 *     Number of rows in R
 * n : &int
 *     Number of columns in R
 * is_col_major : bool
 *     True if R is stored in column-major, false
 *     if R is stored in row-major.
 *
 * Returns
 * -------
 * Nothing, solution is stored in x[].
 *
 * Notes
 * -----
 * R need not be square, the system will be solved over the
 * upper-triangular block of size min(m,n). If remaining entries
 * insolution are unused, they will be set to zero. If a zero
 * is encountered on the ith diagonal, x[i] is set to zero.
 *
 */
template<class I, class T>
void upper_tri_solve(const T R[],
                     const T rhs[],
                     T x[],
                     const I m,
                     const I n,
                     const I is_col_major)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    const I *C;
    if (is_col_major) {
        get_ind = &col_major;
        C = &m;
    }
    else {
        get_ind = &row_major;
        C = &n;
    }

    // Backward substitution
    I rank = std::min(m,n);
    for (I i=(rank-1); i>=0; i--) {
        T temp = rhs[i];
        for (I j=(i+1); j<rank; j++) {
            temp -= R[get_ind(i,j,*C)]*x[j];
        }
        if (std::abs(R[get_ind(i,i,*C)]) < 1e-12) {
            x[i] = 0.0;
        }
        else {
            x[i] = temp / R[get_ind(i,i,*C)];
        }
    }

    // If rank < size of rhs, set free elements in x to zero
    for (I i=m; i<n; i++) {
        x[i] = 0;
    }
}

/* Method to solve the linear least squares problem.
 *
 * Parameters
 * ----------
 * A : double array, length m*n
 *     2d array stored in column- or row-major.
 * b : double array, length m
 *     Right hand side of unconstrained problem.
 * x : double array, length n
 *     Container for solution
 * m : &int
 *     Number of rows in A
 * n : &int
 *     Number of columns in A
 * is_col_major : bool
 *     True if A is stored in column-major, false
 *     if A is stored in row-major.
 *
 * Returns
 * -------
 * x : vector<double>
 *    Solution to constrained least squares problem.
 *
 * Notes
 * -----
 * If system is under determined, free entries are set to zero.
 * Currently only set up for real-valued matrices. May easily
 * generalize to complex, but haven't checked.
 *
 */
template<class I, class T>
void least_squares(T A[],
                   T b[],
                   T x[],
                   const I &m,
                   const I &n,
                   const I is_col_major=0)
{
    // Function pointer for row or column major matrices
    I (*get_ind)(const I, const I, const I);
    if (is_col_major) {
        get_ind = &col_major;
    }
    else {
        get_ind = &row_major;
    }

    // Take QR of A
    std::vector<T> Q = QR(A,m,n,is_col_major);

    // Multiply right hand side, b:= Q^T*b. Have to make new vector, rhs.
    std::vector<T> rhs(m,0);
    for (I i=0; i<m; i++) {
        for (I k=0; k<m; k++) {
            rhs[i] += b[k] * Q[get_ind(k,i,m)];
        }
    }

    // Solve upper triangular system, store solution in x.
    upper_tri_solve(A,&rhs[0],x,m,n,is_col_major);
}

} // namespace pyamg


void fill_colind_and_data(const ParCSRMatrix & A,
                          const ParCSRMatrix &S,
                          const comm_rows & recv_neighbors,
                          const comm_rows &recv_rows,
                          const splitting_t &splitting,
                          const std::vector<int> &cpts,
                          fpoint_distance distance,
                          ParCSRMatrix &R) {
	std::map<int, int> offd_g2l;
	{
		int i{0};
		for (auto col : R.off_proc_column_map)
			offd_g2l[col] = i++;
	}

	auto expand = [&](const int row,
	                  const Matrix & soc,
	                  const auto & split,
	                  Matrix & rmat,
	                  int & ind, auto && colmap) {
		for (int off = soc.idx1[row]; off < soc.idx1[row+1]; ++off) {
			auto d2point = soc.idx2[off];
			if (split[d2point] == Unselected && d2point != row) {
				rmat.idx2[ind++] = std::forward<decltype(colmap)>(colmap)(d2point);
			}
		}
	};
	// build column indices and data for each row of R
	// Note: uses global indices for off_proc columns
	for (std::size_t row = 0; row < cpts.size(); ++row) {
		auto cpoint = cpts[row];
		auto ind = R.on_proc->idx1[row];
		auto ind_off = R.off_proc->idx1[row];
		auto bounds = [=](const Matrix & mat) {
			return std::make_pair(mat.idx1[cpoint], mat.idx1[cpoint + 1]);
		};
		auto & offd_colmap = S.off_proc_column_map;
		// set column indices for R as strongly connected F-points
		auto [beg_on, end_on] = bounds(*S.on_proc);
		for (int off = beg_on; off < end_on; ++off) {
			auto this_point = S.on_proc->idx2[off];
			if (splitting.on_proc[this_point] == Unselected) {
				R.on_proc->idx2[ind++] = this_point;
				// strong distance two F-to-F connections
				if (distance == fpoint_distance::two) {
					expand(this_point, *S.on_proc, splitting.on_proc, *R.on_proc, ind,
					       [](int c) { return c; });
					expand(this_point, *S.off_proc, splitting.off_proc, *R.off_proc, ind_off,
					       [&](int c) { return offd_colmap[c]; });
				}
			}
		}

		auto [beg_off, end_off] = bounds(*S.off_proc);
		for (int off = beg_off; off < end_off; ++off) {
			auto this_point = S.off_proc->idx2[off];
			if (splitting.off_proc[this_point] == Unselected) {
				R.off_proc->idx2[ind_off++] = offd_colmap[this_point];
				if (distance == fpoint_distance::two) {
					recv_neighbors.iter_diag(this_point, [&](auto rview) {
						R.on_proc->idx2[ind++] = rview.idx2();
					});
					recv_neighbors.iter_offd(this_point, [&](auto rview) {
						if (offd_colmap[this_point] != rview.idx2())
							R.off_proc->idx2[ind_off++] = rview.idx2();
					});
				}
			}
		}

		if ((ind != R.on_proc->idx1[row+1] - 1) ||
		    (ind_off != R.off_proc->idx1[row+1])) {
			// int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			std::cout << "ind: " << ind << std::endl;
			std::cout << "rowptr: " << R.on_proc->idx1[row+1] - 1 << std::endl;
			std::cout << "ind_off: " << ind_off << std::endl;
			std::cout << "rowptr_off: " << R.off_proc->idx1[row+1] -1 << std::endl;
			// todo: error checking in raptor
			std::cerr << "Error air_restriction: row pointer does not agree with neighborhood size\n" << std::endl;
		}

        // Build local linear system as the submatrix A restricted to the neighborhood,
        // Nf, of strongly connected F-points to the current C-point, that is A0 =
        // A[Nf, Nf]^T, stored in column major form. Since A in row-major = A^T in
        // column-major, A (CSR) is iterated through and A[Nf,Nf] stored in row-major.
		auto size_n = (ind - R.on_proc->idx1[row]) + (ind_off - R.off_proc->idx1[row]);
		std::vector<double> A0;
		A0.reserve(size_n * size_n);
		auto row_search = [&](int local_row, int global_col) {
			std::optional<double> ret;
			auto search = [&](const Matrix & mat, const std::vector<int> & colmap) {
				for (int off = mat.idx1[local_row]; off < mat.idx1[local_row + 1]; ++off) {
					if (colmap[mat.idx2[off]] == global_col)
						ret.emplace(mat.vals[off]);
				}
			};

			search(*A.on_proc, A.on_proc_column_map);
			search(*A.off_proc, A.off_proc_column_map);

			return ret;
		};
		auto neig_loop = [&](auto && diag, auto && offd) {
			for (int off = R.on_proc->idx1[row]; off < ind; ++off)
				diag(off);
			for (int off = R.off_proc->idx1[row]; off < ind_off; ++off)
				offd(off);
		};
		auto iter_neig = [&](auto && f) {
			auto diag_finder = [&](int i) {
				return [&, i](int col) {
					return row_search(i, col);
				};
			};
			auto offd_finder = [&](int i) {
				return [&, i](int col) {
					std::optional<double> ret;

					auto search = [&](auto rview) {
						if (rview.idx2() == col)
							ret.emplace(rview.val());
					};
					recv_rows.iter_diag(i, search);
					recv_rows.iter_offd(i, search);

					return ret;
				};
			};

			neig_loop(
				[&](int off) {
					auto col = R.on_proc->idx2[off];
					auto global_col = A.on_proc_column_map[col];
					f(global_col, diag_finder(col));
				},
				[&](int off) {
					auto global_col = R.off_proc->idx2[off];
					auto col = offd_g2l[global_col];
					f(global_col, offd_finder(col));
				});
		};
		iter_neig([&](int i, auto && find_in_row){
			iter_neig([&](int j, auto&&) {
				auto maybe_val = find_in_row(j);
				// If index not found, set element to 0
				A0.push_back(maybe_val.value_or(0.));
			});
		});

		/* Build local right hand side given by b_j = -A_{cpt,N_j}, where N_j
		   is the jth indice in the neighborhood of strongly connected F-points
		   to the current C-point. */
		std::vector<double> b0;
		b0.reserve(size_n);
		iter_neig([&](int global_col, auto&&) {
			// Search for indice in row of A. If indice not found, set t0 0.
			auto maybe_val = row_search(cpoint, global_col);
			b0.push_back(-1. * maybe_val.value_or(0.));
		});

        // Solve linear system (least squares solves exactly when full rank)
        // s.t. (RA)_ij = 0 for (i,j) within the sparsity pattern of R. Store
        // solution in data vector for R.
		std::vector<double> sol(size_n);
		if (size_n > 0) {
			constexpr int is_col_major = true;
			pyamg::least_squares(A0.data(), b0.data(), sol.data(), size_n, size_n, is_col_major);

			// fill on_proc and off_proc vals in R
			auto solit = sol.begin();
			neig_loop(
				[&](int off) {
					R.on_proc->vals[off] = *(solit++);
				},
				[&](int off) {
					R.off_proc->vals[off] = *(solit++);
				});

			assert(solit == sol.end());
		}

		// Add identity for C-point in this row
		R.on_proc->idx2[ind] = cpoint;
		R.on_proc->vals[ind] = 1.0;
	}

	// offd colinds are currently global, convert to local
	for (auto & col : R.off_proc->idx2)
		col = offd_g2l[col];
}

/*
  Communicate neighborhood information for distance two f-points.

  For each local row send strongly connected f-point neighbor column indices if
  said row is an f-point. todo: no need to send values.
 */
template<class C>
CSRMatrix * communicate_neighborhood(const ParCSRMatrix & A, const ParCSRMatrix & S,
                                     const splitting_t & split, C && comm) {
	std::vector<int>    rowptr(A.local_num_rows + 1);
	std::vector<int>    colind;
	std::vector<double> values;

	if (A.local_nnz) {
		colind.reserve(A.local_nnz);
		values.reserve(A.local_nnz);
	}

	rowptr[0] = 0;
	for (int i = 0; i < A.local_num_rows; ++i) {
		auto get_bounds = [=](const Matrix & mat) {
			return std::make_pair(mat.idx1[i],
			                      mat.idx1[i+1]);
		};
		auto add_neighborhood = [&](const Matrix & a,
		                            const std::vector<int> & colmap,
		                            const Matrix & s,
		                            const std::vector<int> & splitting) {
			auto [beg, end] = get_bounds(a);
			auto [ctr_s, end_s] = get_bounds(s);

			for (int j = beg; j < end; ++j) {
				int col = a.idx2[j];
				int val = a.vals[j];

				if (splitting[col] == NoNeighbors) continue;

				// add fpoint-fpoint neighborhood
				if (splitting[col] == Unselected &&
				    splitting[i] == Unselected) {
					int global_col = colmap[col];
					// add if strong connection
					if (ctr_s < end_s && s.idx2[ctr_s] == col) {
						colind.push_back(global_col);
						values.push_back(val);
						++ctr_s;
					}
				} else if (ctr_s < end_s && s.idx2[ctr_s] == col)
					++ctr_s;
			}
		};

		add_neighborhood(*A.on_proc, A.on_proc_column_map, *S.on_proc, split.on_proc);
		add_neighborhood(*A.off_proc, A.off_proc_column_map, *S.off_proc, split.off_proc);

		rowptr[i+1] = colind.size();
	}

	return std::forward<C>(comm).communicate(rowptr, colind, values);
}



comm_rows::comm_rows(const ParCSRMatrix &A, CSRMatrix *mat)
	: rmat(mat)
{
	if (!rmat) return;

	/* split remote rows into on and off processor columns.
	   Updates on processor column indices to local id.
	 */
	auto on_proc_partition_to_col = A.map_partition_to_local();
	[&](auto & ... v) {
		((v.resize(rmat->n_rows + 1)), ...);
		((v[0] = 0), ...);
	}(diag.ptr, offd.ptr);

	[&](auto & ... v) {	((v.resize(rmat->nnz)), ...); }(diag.idx, offd.idx);

	int diag_ctr = 0;
	int offd_ctr = 0;
	for (int i = 0; i < rmat->n_rows; ++i) {
		for (int j = rmat->idx1[i]; j < rmat->idx1[i+1]; ++j) {
			int col = rmat->idx2[j];
			if (col < A.partition->first_local_col ||
			    col > A.partition->last_local_col) {
				offd.idx[offd_ctr++] = j;
			} else {
				col = on_proc_partition_to_col[col - A.partition->first_local_row];
				diag.idx[diag_ctr++] = j;
			}
			rmat->idx2[j] = col;
		}
		diag.ptr[i+1] = diag_ctr;
		offd.ptr[i+1] = offd_ctr;
	}

	diag.idx.resize(diag_ctr);
	offd.idx.resize(offd_ctr);
	[](auto & ... v) { ((v.shrink_to_fit()), ...); }(diag.idx, offd.idx);
}

template <class T>
T * compute_R(T & A,
              ParCSRMatrix & S,
              const splitting_t & splitting,
              fpoint_distance distance) {
	auto pre_init = [](auto & mat) {
		mat.sort();
		mat.on_proc->move_diag();
	};
	pre_init(A);
	pre_init(S);

    comm_rows recv_neighbors(A,
                             (distance == fpoint_distance::two) ?
                             communicate_neighborhood(A, S, splitting, *A.comm) : nullptr);

    auto cpts = get_cpoints(splitting.on_proc);
    ParCSRMatrix * R = create_R(A, S, splitting,
                                map_offd_fill_rowptr(
	                                S, splitting, cpts, distance, recv_neighbors));

    comm_rows recv_rows(A, communicate_neighborhood(A, S, splitting, create_neighborhood_comm(*R, A)));

    fill_colind_and_data(A, S, recv_neighbors, recv_rows, splitting, cpts, distance, *R);
    constexpr int tag = 9244;
    R->comm = new ParComm(R->partition,
                          R->off_proc_column_map, R->on_proc_column_map,
                          tag, RAPtor_MPI_COMM_WORLD);

	return R;
}
} // namespace
} // namespace lair

ParCSRMatrix * local_air(ParCSRMatrix & A,
                         ParCSRMatrix & S,
                         const splitting_t & splitting,
                         fpoint_distance distance)
{
	return lair::compute_R(A, S, splitting, distance);
}


ParBSRMatrix * local_air(ParBSRMatrix & A,
                         ParCSRMatrix & S,
                         const splitting_t & splitting,
                         fpoint_distance distance) {
	// return lair::compute_R(A, S, splitting, distance);
	return nullptr;
}

} // namespace raptor
