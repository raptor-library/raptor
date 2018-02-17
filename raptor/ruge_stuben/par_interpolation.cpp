// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

// TODO -- if in S, col is positive, otherwise col is -(col+1)
void communicate(ParCSRMatrix* A, ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, CSRMatrix** recv_on_ptr,
        CSRMatrix** recv_off_ptr)
{
    int start, end, proc;
    int ctr, prev_ctr, size;
    int ctr_S, end_S;
    int row, row_start, row_end;
    int count, row_count, row_size;
    int global_col, col, size_pos;
    int tmp_global_col;
    double val;
    MPI_Status recv_status;
    CommData* recv_comm = A->comm->recv_data;
    CommData* send_comm = A->comm->send_data;
    int key = A->comm->key;
    MPI_Comm mpi_comm = A->comm->mpi_comm;

    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_on = new CSRMatrix(recv_comm->size_msgs, -1);
    CSRMatrix* recv_off = new CSRMatrix(recv_comm->size_msgs, -1);

    // Only sending and recving a single buffer
    struct PairData 
    {
        double val;
        int index;
    };
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    std::vector<int> send_ptr(send_comm->num_msgs+1);
    send_ptr[0] = 0;

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = send_comm->indices[j];
            send_buffer.push_back(PairData());
            size_pos = ctr++; 
            row_start = A->on_proc->idx1[row]+1;
            row_end = A->on_proc->idx1[row+1];
            ctr_S = S->on_proc->idx1[row]+1;
            end_S = S->on_proc->idx1[row+1];

            for (int k = row_start; k < row_end; k++)
            {
                col = A->on_proc->idx2[k];
                if (states[col] == 1)
                {
                    send_buffer.push_back(PairData());
                    send_buffer[ctr].val = A->on_proc->vals[k];
                    global_col = A->on_proc_column_map[col];
                    
                    if (ctr_S < end_S && S->on_proc->idx2[ctr_S] == col)
                    {
                        send_buffer[ctr].index = global_col; // If in S, send positive col
                        ctr_S++;
                    }
                    else
                    {
                        send_buffer[ctr].index = -(global_col+1); // If not in S, store with neg sign

                    }
                    ctr++;

                }
                else if (ctr_S < end_S && S->on_proc->idx2[ctr_S] == col)
                {
                    ctr_S++;
                }

            }

            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            ctr_S = S->off_proc->idx1[row];
            end_S = S->off_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->off_proc->idx2[k];
		if (off_proc_states[col] == -3) continue;
                send_buffer.push_back(PairData());
                send_buffer[ctr].val = A->off_proc->vals[k];
                global_col = A->off_proc_column_map[col];

                if (ctr_S < end_S && S->off_proc_column_map[S->off_proc->idx2[ctr_S]]
                        == A->off_proc_column_map[col])
                {
                    if (off_proc_states[col] == 0) global_col += A->partition->global_num_cols;

                    send_buffer[ctr].index = global_col; // In S, send positive col
                    ctr_S++;
                }
                else
                {
                    if (off_proc_states[col] == 0) global_col += A->partition->global_num_cols;
                    send_buffer[ctr].index = -(global_col+1); // If not in S, store with neg sign
                }

                ctr++;
            }
            send_buffer[size_pos].index = ctr - size_pos - 1;
        }
        send_ptr[i+1] = ctr;
    }
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        proc = send_comm->procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&(send_buffer[start]), end - start, MPI_DOUBLE_INT, proc, 
                key, mpi_comm, &(send_comm->requests[i]));
    }

    row_count = 0;
    recv_on->idx1[0] = 0;
    recv_off->idx1[0] = 0;
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        proc = recv_comm->procs[i];
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        size = end - start;
        MPI_Probe(proc, key, mpi_comm, &recv_status);
        MPI_Get_count(&recv_status, MPI_DOUBLE_INT, &count);
        if (count > recv_buffer.size())
        {
            recv_buffer.resize(count);
        }
        MPI_Recv((&recv_buffer[0]), count, MPI_DOUBLE_INT, proc, key, mpi_comm,
            &recv_status);
        ctr = 0;
        for (int j = 0; j < size; j++)
        {
            row_size = recv_buffer[ctr++].index;
            for (int k = 0; k < row_size; k++)
            {
                val = recv_buffer[ctr].val;
                global_col = recv_buffer[ctr++].index;
                tmp_global_col = global_col;
                if (tmp_global_col < 0)
                    tmp_global_col = (-tmp_global_col) - 1;
                if (tmp_global_col >= A->partition->global_num_cols)
                    tmp_global_col -= A->partition->global_num_cols;

                if (tmp_global_col >= A->partition->first_local_row &&
                        tmp_global_col <= A->partition->last_local_row)
                {
                    recv_on->idx2.push_back(global_col);
                    recv_on->vals.push_back(val);
                }
                else
                {
                    if (global_col >= A->partition->global_num_cols || 
                            global_col < -(A->partition->global_num_cols))
                            continue; // Don't add Fine points to off proc
                    recv_off->idx2.push_back(global_col);
                    recv_off->vals.push_back(val);
                }
            }
            recv_on->idx1[row_count+1] = recv_on->idx2.size();
            recv_off->idx1[row_count+1] = recv_off->idx2.size();
            row_count++;
        }
    }
    recv_on->nnz = recv_on->idx2.size();
    recv_off->nnz = recv_off->idx2.size();

    MPI_Waitall(send_comm->num_msgs, send_comm->requests.data(), MPI_STATUSES_IGNORE);

    *recv_on_ptr = recv_on;
    *recv_off_ptr = recv_off;
}


void communicate(ParCSRMatrix* A, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, CSRMatrix** recv_on_ptr,
        CSRMatrix** recv_off_ptr)
{
    int start, end, proc;
    int ctr, prev_ctr, size;
    int row, row_start, row_end;
    int count, row_count, row_size;
    int global_col, col, size_pos;
    double val;
    MPI_Status recv_status;
    CommData* recv_comm = A->comm->recv_data;
    CommData* send_comm = A->comm->send_data;
    int key = A->comm->key;
    MPI_Comm mpi_comm = A->comm->mpi_comm;

    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_on = new CSRMatrix(recv_comm->size_msgs, -1);
    CSRMatrix* recv_off = new CSRMatrix(recv_comm->size_msgs, -1);

    // Only sending and recving a single buffer
    struct PairData 
    {
        double val;
        int index;
    };
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    std::vector<int> send_ptr(send_comm->num_msgs+1);
    send_ptr[0] = 0;

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = send_comm->indices[j];
            send_buffer.push_back(PairData());
            size_pos = ctr++; 
            row_start = A->on_proc->idx1[row];
            row_end = A->on_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->on_proc->idx2[k];
                if (states[col] == 1)
                {
                    send_buffer.push_back(PairData());
                    send_buffer[ctr].index = A->on_proc_column_map[col];
                    send_buffer[ctr++].val = A->on_proc->vals[k];
                }
            }
            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->off_proc->idx2[k];
                if (off_proc_states[col] == 1)
                {
                    send_buffer.push_back(PairData());
                    send_buffer[ctr].index = A->off_proc_column_map[col];
                    send_buffer[ctr++].val = A->off_proc->vals[k];
                }
            }
            send_buffer[size_pos].index = ctr - size_pos - 1;
        }
        send_ptr[i+1] = ctr;
    }
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        proc = send_comm->procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&(send_buffer[start]), end - start, MPI_DOUBLE_INT, proc, 
                key, mpi_comm, &(send_comm->requests[i]));
    }

    row_count = 0;
    recv_on->idx1[0] = 0;
    recv_off->idx1[0] = 0;
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        proc = recv_comm->procs[i];
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        size = end - start;
        MPI_Probe(proc, key, mpi_comm, &recv_status);
        MPI_Get_count(&recv_status, MPI_DOUBLE_INT, &count);
        if (count > recv_buffer.size())
        {
            recv_buffer.resize(count);
        }
        MPI_Recv((&recv_buffer[0]), count, MPI_DOUBLE_INT, proc, key, mpi_comm,
            &recv_status);
        ctr = 0;
        for (int j = 0; j < size; j++)
        {
            row_size = recv_buffer[ctr++].index;
            for (int k = 0; k < row_size; k++)
            {
                global_col = recv_buffer[ctr].index;
                val = recv_buffer[ctr++].val;
                if (global_col >= A->partition->first_local_row &&
                        global_col <= A->partition->last_local_row)
                {
                    recv_on->idx2.push_back(global_col);
                    recv_on->vals.push_back(val);
                }
                else
                {
                    recv_off->idx2.push_back(global_col);
                    recv_off->vals.push_back(val);
                }
            }
            recv_on->idx1[row_count+1] = recv_on->idx2.size();
            recv_off->idx1[row_count+1] = recv_off->idx2.size();
            row_count++;
        }
    }
    recv_on->nnz = recv_on->idx2.size();
    recv_off->nnz = recv_off->idx2.size();

    MPI_Waitall(send_comm->num_msgs, send_comm->requests.data(), MPI_STATUSES_IGNORE);

    *recv_on_ptr = recv_on;
    *recv_off_ptr = recv_off;
}

ParCSRMatrix* extended_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, 
        CommPkg* comm)
{
    int start, end, idx;
    int start_S, end_S;
    int start_k, end_k;
    int col, global_col;
    int ctr, col_k, col_P;
    int col_S, col_A;
    int global_num_cols;
    int on_proc_cols, off_proc_cols;
    int sign;
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    double coarse_sum;
    double val, weak_sum;

    std::set<int> global_set;
    std::map<int, int> global_to_local;
    std::vector<int> off_proc_column_map;

    CSRMatrix* recv_on; // On Proc Block of Recvd A
    CSRMatrix* recv_off; // Off Proc Block of Recvd A
    CSRMatrix* recv_on_S;
    CSRMatrix* recv_off_S;

    A->sort();
    S->sort();
    A->on_proc->move_diag();
    S->on_proc->move_diag();

    // Gather off_proc_states_A
    std::vector<int> off_proc_states_A;
    if (A->off_proc_num_cols) off_proc_states_A.resize(A->off_proc_num_cols);
    std::vector<int>& recvbuf = comm->communicate(states);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        off_proc_states_A[i] = recvbuf[i];
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
    communicate(A, S, states, off_proc_states_A, &recv_on, &recv_off);

    // Change on_proc_cols to local
    recv_on->n_cols = A->on_proc_num_cols;
    int* on_proc_partition_to_col = A->map_partition_to_local();
    for (std::vector<int>::iterator it = recv_on->idx2.begin();
            it != recv_on->idx2.end(); ++it)
    {
        if (*it >= 0)
        {
            // In S, add positive column
            if (*it >= A->partition->global_num_cols)
            {
                *it -= A->partition->global_num_cols;
                *it = on_proc_partition_to_col[*it - A->partition->first_local_row];
                *it += A->partition->global_num_cols;
            }
            else
            {
                *it = on_proc_partition_to_col[*it - A->partition->first_local_row];
            }
        }
        else
        {
            // Not in S, add negative column
            global_col = (-*it)-1;

            if (global_col >= A->partition->global_num_cols)
            {
                global_col -= A->partition->global_num_cols;
                global_col = on_proc_partition_to_col[global_col - A->partition->first_local_row];
                global_col += A->partition->global_num_cols;
            }
            else
            {
                global_col = on_proc_partition_to_col[global_col - A->partition->first_local_row];
            }
            *it  = -(global_col + 1);
        }
    }
    delete[] on_proc_partition_to_col;

    // Change off_proc_cols to local (remove cols not on rank)
    off_proc_cols = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == 1)
        {
            off_proc_cols++;
            global_set.insert(S->off_proc_column_map[i]);
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == 0)
        {
            col_A = off_proc_S_to_A[i];
            start = recv_off->idx1[col_A];
            end = recv_off->idx1[col_A+1];
            for (int j = start; j < end; j++)
            {
                global_col = recv_off->idx2[j];
                if (global_col < 0)
                    global_col = - (global_col - 1);
                std::set<int>::iterator it = global_set.find(global_col);
                if (it == global_set.end())
                {
                    global_set.insert(global_col);
                    off_proc_cols++; // Recv off has only coarse points
                }
            }
        }
    }
    for (std::set<int>::iterator it = global_set.begin(); it != global_set.end(); ++it)
    {
        global_to_local[*it] = off_proc_column_map.size();
        off_proc_column_map.push_back(*it);
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
            sign = 1.0;
            if (global_col < 0)
            {
                sign = -1.0;
                global_col = (-global_col) - 1;
            }
            std::map<int, int>::iterator it = global_to_local.find(global_col);
            if (it != global_to_local.end())
            {
                if (sign > 0)
                {
                    // In S, add positive column
                    recv_off->idx2[ctr] = it->second;
                }
                else
                {
                    // Not in S, make column negative
                    recv_off->idx2[ctr] = -(it->second + 1);
                }
                recv_off->vals[ctr++] = recv_off->vals[j];
            }
        }
        recv_off->idx1[i+1] = ctr;
        start = end;
    }
    recv_off->nnz = ctr;
    recv_off->idx2.resize(ctr);
    recv_off->vals.resize(ctr);

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
        if (states[i] == 1)
        {
            on_proc_cols++;
        }
    }
    MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
    ParCSRMatrix* P = new ParCSRMatrix(A->partition, A->global_num_rows, global_num_cols, 
            A->local_num_rows, on_proc_cols, off_proc_cols);

    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == 1)
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
        if (off_proc_states[i] != 1)
        {
            continue; // Only for coarse points
        }

        col_A = off_proc_S_to_A[i];
        while (off_proc_column_map[ctr] < A->off_proc_column_map[col_A])
        {
            ctr++;
        }
        off_proc_A_to_P[col_A] = ctr;
    }
    

    // For each row, will calculate coarse sums and store 
    // strong connections in vector
    std::vector<int> pos;
    std::vector<int> off_proc_pos;
    std::vector<int> recv_off_pos;
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
        off_proc_row_strong.resize(A->off_proc_num_cols, 0);
    }
    if (P->off_proc_num_cols)
    {
        off_proc_pos.resize(P->off_proc_num_cols, -1);
        off_proc_row_coarse.resize(P->off_proc_num_cols, 0);
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
        weak_sum = A->on_proc->vals[start-1];
        row_start_on = P->on_proc->idx2.size();
        row_start_off = P->off_proc->idx2.size();
        row_coarse[i] = 1;

        if (weak_sum < 0)
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
                    row_coarse[col] = 1;               
                }
                else if (states[col] == 0)
                {
                    row_strong[col] = val;
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
                if (off_proc_states_A[col] == 1)
                {
                    col_P = off_proc_A_to_P[col];
                    off_proc_pos[col_P] = P->off_proc->idx2.size();
                    col_exists[col_P] = true;
                    P->off_proc->idx2.push_back(col_P);
                    P->off_proc->vals.push_back(val);
                    off_proc_row_coarse[col_P] = 1;
                }
                else if (off_proc_states_A[col] == 0)
                {
                    off_proc_row_strong[col] = val;
                }
                ctr++;
            }
            else
            {
                weak_sum += val;
            }
        }

        // Add distance-2 processes to row_coarse
        start = S->on_proc->idx1[i]+1;
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            if (states[col] == 0)
            {
                start_k = S->on_proc->idx1[col]+1;
                end_k = S->on_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = S->on_proc->idx2[k];
                    if (states[col_k] == 1 && row_coarse[col_k] == 0)
                    {
                        pos[col_k] = P->on_proc->idx2.size();
                        P->on_proc->idx2.push_back(on_proc_col_to_new[col_k]);
                        P->on_proc->vals.push_back(0.0); // Aij = 0
                        row_coarse[col_k] = 1;
                    }
                }
                start_k = S->off_proc->idx1[col];
                end_k = S->off_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = S->off_proc->idx2[k];
                    if (off_proc_states[col_k] == 1)
                    {
                        col_A = off_proc_S_to_A[col_k];
                        col_P = off_proc_A_to_P[col_A];
                        if (off_proc_row_coarse[col_P] == 0)
                        {
                            off_proc_pos[col_P] = P->off_proc->idx2.size();
                            col_exists[col_P] = true;
                            P->off_proc->idx2.push_back(col_P);
                            P->off_proc->vals.push_back(0.0); // Aij = 0
                            off_proc_row_coarse[col_P] = 1;
                        }
                    }
                }
            }
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];
            if (off_proc_states[col] == 0)
            {
                col_A = off_proc_S_to_A[col];
                start_k = recv_on->idx1[col_A];
                end_k = recv_on->idx1[col_A+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = recv_on->idx2[k];
                    if (col_k < 0 || col_k >= A->partition->global_num_cols)
                    {
                        continue; // NOT IN S
                    }
                    if (row_coarse[col_k] == 0)
                    {
                        pos[col_k] = P->on_proc->idx2.size();
                        P->on_proc->idx2.push_back(on_proc_col_to_new[col_k]);
                        P->on_proc->vals.push_back(0.0);
                        row_coarse[col_k] = 1;
                    }
                }
                start_k = recv_off->idx1[col_A];
                end_k = recv_off->idx1[col_A+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = recv_off->idx2[k];
                    if (col_k < 0) 
                    {
                        continue; // NOT IN S
                    }
                    if (off_proc_row_coarse[col_k] == 0)
                    {
                        off_proc_pos[col_k] = P->off_proc->idx2.size();
                        col_exists[col_k] = true;
                        P->off_proc->idx2.push_back(col_k);
                        P->off_proc->vals.push_back(0.0);
                        off_proc_row_coarse[col_k] = 1;
                    }
                }
            }
        }

        row_end_on = P->on_proc->idx2.size();
        row_end_off = P->off_proc->idx2.size();

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
                        col_k = A->off_proc->idx2[k]; 
                        col_P = off_proc_A_to_P[col_k];
                        if (col_P >= 0)
                        {
                            val = A->off_proc->vals[k] * off_proc_row_coarse[col_P];
                            if (val * sign < 0)
                            {
                                coarse_sum += val;
                            }
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
                        if (col_k < 0)
                            col_k = (-col_k) - 1;
                        if (col_k >= A->partition->global_num_cols)
                        {
                            col_k -= A->partition->global_num_cols;
                            if (col_k != i) continue;
                        }

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
                        col_k = recv_off->idx2[k]; // Already in col_P
                        if (col_k < 0)
                        {
                            col_k = (-col_k) - 1;
                        }
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
            if (states[col] == 0) // k in F_i^S
            {
                start_k = A->on_proc->idx1[col]+1;
                end_k = A->on_proc->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = A->on_proc->idx2[k];
                    val = A->on_proc->vals[k];
                    idx = pos[col_k];
                    if (val * sign < 0 && col_k == i)
                    {
                        weak_sum += (row_strong[col] * val);
                    }
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
                    col_P = off_proc_A_to_P[col_k];
                    if (col_P >= 0)
                    {
                        val = A->off_proc->vals[k];
                        idx = off_proc_pos[col_P];
                        if (val * sign < 0 && idx >= 0)
                        {
                            P->off_proc->vals[idx] += (row_strong[col] * val);
                        }
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
                    if (col_k < 0)
                    {
                        col_k = (-col_k) - 1;
                    }

                    if (val * sign < 0 && col_k - A->partition->global_num_cols == i)
                    {
                        weak_sum += (off_proc_row_strong[col] * val);
                    }
                    if (col_k >= A->partition->global_num_cols) continue;
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
                    if (col_k < 0)
                    {
                        col_k = (-col_k) - 1;
                    }
                    idx = off_proc_pos[col_k];
                    if (val * sign < 0 && idx >= 0)
                    {
                        P->off_proc->vals[idx] += (off_proc_row_strong[col] * val);
                    }
                }
            }
        }


        // Divide by weak sum and clear row values
        for (int j = row_start_on; j < row_end_on; j++)
        {
            col = P->on_proc->idx2[j];
            P->on_proc->vals[j] /= -weak_sum;
            pos[col] = -1;
            row_coarse[col] = 0;
        }
        for (int j = row_start_off; j < row_end_off; j++)
        {
            col = P->off_proc->idx2[j];
            P->off_proc->vals[j] /= -weak_sum;
            off_proc_pos[col] = -1;
            off_proc_row_coarse[col] = 0;
        }
        row_coarse[i] = 0;

        for (int j = 0; j < A->on_proc_num_cols; j++)
        {
            row_coarse[j] = 0;
            pos[j] = -1;
        }
        for (int j = 0; j < P->off_proc_num_cols; j++)
        {
            off_proc_row_coarse[j] = 0;
            off_proc_pos[j] = -1;
        }

        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            row_strong[col] = 0;
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = off_proc_S_to_A[S->off_proc->idx2[j]];
            off_proc_row_strong[col] = 0;
        }
        P->on_proc->idx1[i+1] = P->on_proc->idx2.size();
        P->off_proc->idx1[i+1] = P->off_proc->idx2.size();
    }
    P->on_proc->nnz = P->on_proc->idx2.size();
    P->off_proc->nnz = P->off_proc->idx2.size();
    P->local_nnz = P->on_proc->nnz + P->off_proc->nnz;

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

    if (S->comm)
    {
        P->comm = new ParComm(P->partition, P->off_proc_column_map,
                P->on_proc_column_map);
    }

    if (S->tap_comm)
    {
        P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map,
                P->on_proc_column_map);
    }
    
    delete recv_on;
    delete recv_off;

    return P;
}

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states, 
        CommPkg* comm)
{
    int start, end;
    int start_k, end_k;
    int end_S;
    int col, col_k, col_S;
    int ctr;
    int global_col;
    int global_num_cols;
    double diag, val;
    double weak_sum, coarse_sum;
    double sign;

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
    MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
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
    std::vector<int>& recvbuf = comm->communicate(states);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        off_proc_states_A[i] = recvbuf[i];
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
    communicate(A, states, off_proc_states_A, &recv_on, &recv_off);

    // Change on_proc_cols to local
    recv_on->n_cols = A->on_proc_num_cols;
    int* on_proc_partition_to_col = A->map_partition_to_local();
    for (std::vector<int>::iterator it = recv_on->idx2.begin();
            it != recv_on->idx2.end(); ++it)
    {
        *it = on_proc_partition_to_col[*it - A->partition->first_local_row];
    }
    delete[] on_proc_partition_to_col;

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

    if (S->tap_comm)
    {
        P->tap_comm = new TAPComm(S->tap_comm, on_proc_col_to_new,
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
    MPI_Allreduce(&(on_proc_cols), &global_num_cols, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
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

    if (S->tap_comm)
    {
        S->tap_comm = new TAPComm(S->tap_comm, on_proc_col_to_new,
                off_proc_col_to_new);
    }

    return P;
}





