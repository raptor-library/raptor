// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_cf_splitting.hpp"

using namespace raptor;

void binary_transpose(const ParCSRMatrix* S,
        std::vector<int>& on_col_ptr, 
        std::vector<int>& off_col_ptr, 
        std::vector<int>& on_col_indices,
        std::vector<int>& off_col_indices)
{
    int start, end;
    int col, idx;
    std::vector<int> on_col_sizes;
    std::vector<int> off_col_sizes;

    // Resize to corresponding dimensions of S
    on_col_ptr.resize(S->on_proc_num_cols+1);
    off_col_ptr.resize(S->off_proc_num_cols+1);
    if (S->on_proc_num_cols)
    {
        on_col_sizes.resize(S->on_proc_num_cols, 0);
        on_col_indices.resize(S->on_proc->nnz);
    }
    if (S->off_proc_num_cols)
    {
        off_col_sizes.resize(S->off_proc_num_cols, 0);
        off_col_indices.resize(S->off_proc->nnz);
    }

    // Calculate nnz in each col
    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            on_col_sizes[col]++;
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];
            off_col_sizes[col]++;
        }
    }

    // Create col_ptrs
    on_col_ptr[0] = 0;
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        on_col_ptr[i+1] = on_col_ptr[i] + on_col_sizes[i];
        on_col_sizes[i] = 0;
    }
    off_col_ptr[0] = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        off_col_ptr[i+1] = off_col_ptr[i] + off_col_sizes[i];
        off_col_sizes[i] = 0;
    }

    // Add indices to col_indices
    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            idx = on_col_ptr[col] + on_col_sizes[col]++;
            on_col_indices[idx] = i;
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];
            idx = off_col_ptr[col] + off_col_sizes[col]++;
            off_col_indices[idx] = i;
        }
    }
}

void initial_cljp_weights(const ParCSRMatrix* S,
        const std::vector<int>& col_ptr,
        const std::vector<int>& col_indices,
        const std::vector<int>& states,
        std::vector<double>& weights,
        std::vector<int>& on_edgemark)
{
    int start, end;
    int idx, idx_k;
    int idx_start, idx_end;
    int head, length;
    int proc;
    int tag = 3029;
    int tmp;

    std::vector<int> recv_weights;
    std::vector<int> off_proc_weights;
    std::vector<int> dist2;
    std::vector<int> next;

    if (S->off_proc_num_cols)
    {
        off_proc_weights.resize(S->off_proc_num_cols, 0);
    }
    if (S->comm->send_data->size_msgs)
    {
        recv_weights.resize(S->comm->send_data->size_msgs);
    }
    if (S->local_num_rows)
    {
        dist2.resize(S->local_num_rows, 0);
        next.resize(S->local_num_rows, -1);
    }

    // Set each weight initially to random value [0,1)
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        // Seed random generator with global col
        srand(S->on_proc_column_map[i]);

        // Random value [0,1)
        weights[i] = rand();
        if (weights[i] == RAND_MAX)
            weights[i] -= 1;
        weights[i] /= RAND_MAX;
    }

    // Go through each row i, for each column j
    // add to weights or off_proc_weights at j
    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            weights[idx] += 1;
        }

        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->off_proc->idx2[j];
            off_proc_weights[idx] += 1;
        }
    }
    // Send off_proc_weights to neighbors
    for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
    {
        proc = S->comm->recv_data->procs[i];
        start = S->comm->recv_data->indptr[i];
        end = S->comm->recv_data->indptr[i+1];
        MPI_Issend(&(off_proc_weights[start]), end - start, MPI_INT, proc,
                tag, MPI_COMM_WORLD, &(S->comm->recv_data->requests[i]));
    }

    // Recv weights from neighbors and add to local weights
    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        proc = S->comm->send_data->procs[i];
        start = S->comm->send_data->indptr[i];
        end = S->comm->send_data->indptr[i+1];
        MPI_Irecv(&(recv_weights[start]), end - start, MPI_INT, proc, tag, 
                MPI_COMM_WORLD, &(S->comm->send_data->requests[i]));
    }

    // Wait for recvs to complete
    if (S->comm->send_data->num_msgs)
    {
        MPI_Waitall(S->comm->send_data->num_msgs, 
                S->comm->send_data->requests.data(),
                MPI_STATUSES_IGNORE);
    }

    // Update weights based on recv_weights
    for (int i = 0; i < S->comm->send_data->size_msgs; i++)
    {
        idx = S->comm->send_data->indices[i];
        weights[idx] += recv_weights[i];
    }
    
    // Wait for sends to complete
    if (S->comm->recv_data->num_msgs)
    {
        MPI_Waitall(S->comm->recv_data->num_msgs, 
                S->comm->recv_data->requests.data(),
                MPI_STATUSES_IGNORE);
    }


    // Update neighbors of coarse interior points
    for (int i = 0; i < S->local_num_rows; i++)
    {
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }

        if (states[i] == 1)
        {
            // Subtract weight from neighbors of coarse
            for (int j = start; j < end; j++)
            {
                // Subtract weight only if connection
                // was not already checked (Sij still 1)
                if (on_edgemark[j])
                {
                    idx = S->on_proc->idx2[j];
                    weights[idx] -= 1;
                    on_edgemark[j] = 0;
                }
            }
        }
        else
        {
            // Update Sij for all coarse j in row
            for (int j = start; j < end; j++)
            {
                idx = S->on_proc->idx2[j];
                if (states[idx] == 1)
                {
                    on_edgemark[j] = 0;
                }
            }
        }
    }
    // Update dist2 neighbors of coarse interior points
    for (int i = 0; i < S->local_num_rows; i++)
    {
        head = -2;
        length = 0;

        // Find any unassigned dist2 indices, sharing 
        // a coarse point with row i (Sik != 0 and 
        // Sjk != 0 for all j in dist2)
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (states[idx] == 1)
            {
                idx_start = col_ptr[idx];
                idx_end = col_ptr[idx+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx_k = col_indices[k];
                    if (states[idx_k] == -1)
                    {
                        dist2[idx_k] = 1;
                        if (next[idx_k] == -1)
                        {
                            next[idx_k] = head;
                            head = idx_k;
                            length++;
                        }
                    }
                }
            }
        }

        // Go through row and see if there is any col j
        // in dist2 such that Sij = 1
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (on_edgemark[j] && dist2[idx])
            {
                weights[idx] -= 1;
                on_edgemark[j] = 0;
            }
        }

        // Remove entries from dist2 and next
        for (int j = 0; j < length; j++)
        {
            tmp = head;
            head = next[tmp];
            dist2[tmp] = 0;
            next[tmp] = -1;
        }
    }

    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] != -1)
        {
            weights[i] = 0;
        }
    }
}

void update_weights(ParCSRMatrix* S,
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        std::vector<double>& weights,
        std::vector<int>& off_proc_weight_updates,
        std::vector<int>& on_edgemark,
        std::vector<int>& off_edgemark)
{
    int start_on, end_on;
    int start_off, end_off;
    int idx;
    
    for (int i = 0; i < S->local_num_rows; i++)
    {        
        start_on = S->on_proc->idx1[i];
        end_on = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start_on] == i)
        {
            start_on++;
        }
        start_off = S->off_proc->idx1[i];
        end_off = S->off_proc->idx1[i+1];

        if (states[i] == 2)
        {
            for (int j = start_on; j < end_on; j++)
            {
                if (on_edgemark[j])
                {
                    idx = S->on_proc->idx2[j];
                    weights[idx] -= 1;
                    on_edgemark[j] = -1;
                }
            }
            for (int j = start_off; j < end_off; j++)
            {
                if (off_edgemark[j] > 0)
                {
                    idx = S->off_proc->idx2[j];
                    off_proc_weight_updates[idx] -= 1;
                    off_edgemark[j] = -1;
                }
            }
        }
        else
        {
            for (int j = start_on; j < end_on; j++)
            {
                idx = S->on_proc->idx2[j];
                if (states[idx] == 2)
                {
                    on_edgemark[j] = -1;
                }
            }
            for (int j = start_off; j < end_off; j++)
            {
                idx = S->off_proc->idx2[j];
                if (off_proc_states[idx] == 2)
                {
                    off_edgemark[j] = -1;
                }
            }
        }
    }
}

void update_weights_onproc_dist2(const ParCSRMatrix* S,
        const std::vector<int>& on_col_ptr,
        const std::vector<int>& off_col_ptr,
        const std::vector<int>& on_col_indices,
        const std::vector<int>& off_col_indices,
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        std::vector<double>& weights,
        std::vector<int>& on_edgemark)
{
    int start, end;
    int head, length;
    int idx, idx_k, tmp;
    int idx_start, idx_end;

    std::vector<int> dist2;
    std::vector<int> next;

    if (S->local_num_rows)
    {
        dist2.resize(S->local_num_rows, 0);
        next.resize(S->local_num_rows, -1);
    }
    
    // Update dist2 neighbors of coarse interior points
    for (int i = 0; i < S->local_num_rows; i++)
    {
        head = -2;
        length = 0;

        // Find any unassigned dist2 indices, sharing 
        // a coarse point with row i (Sik != 0 and 
        // Sjk != 0 for all j in dist2)
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (states[idx] == 2)
            {
                idx_start = on_col_ptr[idx];
                idx_end = on_col_ptr[idx+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx_k = on_col_indices[k];
                    if (states[idx_k] == -1)
                    {
                        dist2[idx_k] = 1;
                        if (next[idx_k] == -1)
                        {
                            next[idx_k] = head;
                            head = idx_k;
                            length++;
                        }
                    }
                }
            }
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->off_proc->idx2[j];
            if (off_proc_states[idx] == 2)
            {
                idx_start = off_col_ptr[idx];
                idx_end = off_col_ptr[idx+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx_k = off_col_indices[k];
                    if (states[idx_k] == -1)
                    {
                        dist2[idx_k] = 1;
                        if (next[idx_k] == -1)
                        {
                            next[idx_k] = head;
                            head = idx_k;
                            length++;
                        }
                    }
                }
            }
        }

        // Go through row and see if there is any col j
        // in dist2 such that Sij = 1
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (on_edgemark[j] && dist2[idx])
            {
                weights[idx] -= 1;
                on_edgemark[j] = -1;
            }
        }

        // Remove entries from dist2 and next
        for (int j = 0; j < length; j++)
        {
            tmp = head;
            head = next[tmp];
            dist2[tmp] = 0;
            next[tmp] = -1;
        }
    }
}

void update_weights_offproc_dist2(const ParCSRMatrix* S,
        const std::vector<int>& off_proc_col_ptr,
        const std::vector<int>& off_proc_col_coarse,
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        std::vector<int>& off_proc_weight_updates, 
        std::vector<int>& off_edgemark)
{
    int head, length;
    int start, end;
    int idx, idx_k;
    int idx_start, idx_end;
    int off_col, tmp;
    int n_cols;


    std::vector<int> row_coarse;
    std::vector<int> next;
    n_cols = S->on_proc_num_cols + S->off_proc_num_cols;
    if (n_cols)
    {
        row_coarse.resize(n_cols, 0);
        next.resize(n_cols);
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        head = -2;
        length = 0;

        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (states[idx] == 2)
            {
                row_coarse[idx] = 1;
                next[idx] = head;
                head = idx;
                length++;
            }
        }
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->off_proc->idx2[j];
            if (off_proc_states[idx] == 2)
            {
                off_col = idx + S->on_proc_num_cols;
                row_coarse[off_col] = 1;
                next[off_col] = head;
                head = off_col;
                length++;
            }
        }
        for (int j = start; j < end; j++)
        {
            idx = S->off_proc->idx2[j];

            // Check if unassigned and Sij == 1
            if (off_proc_states[idx] == -1 && off_edgemark[j] > 0)
            {
                // Check if any coarse cols match (Sik != 0 && Sjk != 0)
                idx_start = off_proc_col_ptr[idx];
                idx_end = off_proc_col_ptr[idx+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx_k = off_proc_col_coarse[k];

                    // If there is a match, decrease weight,
                    // set Sij to -1, and break the loop
                    if (row_coarse[idx_k])
                    {
                        off_proc_weight_updates[idx] -= 1;
                        off_edgemark[j] = -1;
                        break;
                    }
                }
            }
        }

        for (int j = 0; j < length; j++)
        {
            row_coarse[head] = 0;
            head = next[head];
        }
    }
}

void find_off_proc_weights(const ParCSRMatrix* S, 
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        const std::vector<double>& weights,
        std::vector<double>& off_proc_weights)
{
    int start, end;
    int idx, proc, size;
    int ctr = 0;
    int prev_ctr = 0;
    int n_sends = 0;
    int n_recvs = 0;
    int tag = 1992;

    std::vector<int> recv_indices;
    if (S->comm->recv_data->size_msgs)
    {
        recv_indices.resize(S->comm->recv_data->size_msgs);
    }

    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        proc = S->comm->send_data->procs[i];
        start = S->comm->send_data->indptr[i];
        end = S->comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->comm->send_data->indices[j];
            if (states[idx] == -1)
            {
                S->comm->send_data->buffer[ctr++] = weights[idx];
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Issend(&(S->comm->send_data->buffer[prev_ctr]), size, MPI_DOUBLE, 
                    proc, tag, MPI_COMM_WORLD, &(S->comm->send_data->requests[n_sends++]));
            prev_ctr = ctr;
        }
    }

    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
    {
        proc = S->comm->recv_data->procs[i];
        start = S->comm->recv_data->indptr[i];
        end = S->comm->recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            if (off_proc_states[j] == -1)
            {
                recv_indices[ctr++] = j;
            }
            else
            {
                off_proc_weights[j] = 0;
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Irecv(&(S->comm->recv_data->buffer[prev_ctr]), size, MPI_DOUBLE,
                    proc, tag, MPI_COMM_WORLD, &(S->comm->recv_data->requests[n_recvs++]));
            prev_ctr = ctr;
        }
    }

    if (n_sends)
    {
        MPI_Waitall(n_sends, S->comm->send_data->requests.data(), MPI_STATUSES_IGNORE);
    }
    if (n_recvs)
    {
        MPI_Waitall(n_recvs, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < ctr; i++)
    {
        idx = recv_indices[i];
        off_proc_weights[idx] = S->comm->recv_data->buffer[i];
    }
}

void find_max_off_weights(const ParCSRMatrix* S,
        const std::vector<int>& off_col_ptr,
        const std::vector<int>& off_col_indices,
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        const std::vector<double>& weights,
        std::vector<double>& max_weights)
{
    int start, end;
    int idx_start, idx_end;
    int idx;
    int proc;
    double max_weight;
    int tag = 3266;
    int ctr, prev_ctr, size;
    int n_sends = 0;
    int n_recvs = 0;

    std::vector<int> recv_indices;
    if (S->comm->send_data->size_msgs)
    {
        recv_indices.resize(S->comm->send_data->size_msgs);
    }

    // Send max off_proc col weights to appropriate proc
    // Only find / send max weight in col if 
    // off_proc_states[col] == -1
    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
    {
        proc = S->comm->recv_data->procs[i];
        start = S->comm->recv_data->indptr[i];
        end = S->comm->recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            if (off_proc_states[j] == -1)
            {
                max_weight = 0;
                idx_start = off_col_ptr[j];
                idx_end = off_col_ptr[j+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx = off_col_indices[k];
                    if (weights[idx] > max_weight)
                    {
                        max_weight = weights[idx];
                    }
                }
                S->comm->recv_data->buffer[ctr++] = max_weight;
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Issend(&(S->comm->recv_data->buffer[prev_ctr]), size, MPI_DOUBLE, proc, 
                tag, MPI_COMM_WORLD, &(S->comm->recv_data->requests[n_sends++]));
            prev_ctr = ctr;
        }
    }

    // Recv max col weights associated with local indices
    // Store max recvd in max_weights
    // Only recv max weight if states[idx] == -1
    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        proc = S->comm->send_data->procs[i];
        start = S->comm->send_data->indptr[i];
        end = S->comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->comm->send_data->indices[j];
            if (states[idx] == -1)
            {
                recv_indices[ctr++] = idx;
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Irecv(&(S->comm->send_data->buffer[prev_ctr]), size, MPI_DOUBLE, proc,
                    tag, MPI_COMM_WORLD, &(S->comm->send_data->requests[n_recvs++]));
            prev_ctr = ctr;
        }
    }

    // Wait for recvs to complete
    if (n_recvs)
    {
        MPI_Waitall(n_recvs, S->comm->send_data->requests.data(), MPI_STATUSES_IGNORE);
    }

    // Find max recvd weights associated with each local idx
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        max_weights[i] = 0;
    }
    for (int i = 0; i < ctr; i++)
    {
        idx = recv_indices[i];
        if (S->comm->send_data->buffer[i] > max_weights[idx])
        {
            max_weights[idx] = S->comm->send_data->buffer[i];
        }
    }

    // Wait for sends to complete
    if (n_sends)
    {
        MPI_Waitall(n_sends, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
    }
}

int select_independent_set(const ParCSRMatrix* S, 
        const std::vector<double>& weights, 
        const std::vector<double>& off_proc_weights,
        const std::vector<double>& max_off_weights,
        const std::vector<int>& on_col_ptr,
        const std::vector<int>& on_col_indices,
        std::vector<int>& states)
{
    int start, end;
    int col;
    int j;
    int num_new_coarse = 0;
    double weight;

    for (int i = 0; i < S->local_num_rows; i++)
    {
        // Only need to update unassigned states
        if (states[i] != -1) 
            continue;
            
        // Check if weight of i is greater than all weights influenced by i
        weight = weights[i];
        
        // Check if max_off_weights[i] is greater than i's weight
        // (max_off_weights contains max weight in column over all other procs)
        if (max_off_weights[i] > weight)
            continue;

        // Check if row in on_proc has greater max weight
        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        if (S->on_proc->idx2[start] == i)
        {
            start++;
        }
        for (j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            if (weights[col] > weight)
                break;
        }
        if (j != end)
            continue;

        // Check if row in off_proc has greater max weight
        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        for (j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];

            if (off_proc_weights[col] > weight)
                break;
        }
        if (j != end)
            continue;

        // Check if on_proc col has greater max weight
        start = on_col_ptr[i];
        end = on_col_ptr[i+1];
        for (j = start; j < end; j++)
        {
            col = on_col_indices[j];
            if (weights[col] > weight)
                break;
        }
        if (j != end)
            continue;

        // If made it this far, weight is greater than all unassigned neighbors
        states[i] = 2;
        num_new_coarse++;
    }

    return num_new_coarse;
}

int find_off_proc_states(const ParCSRMatrix* S,
        const std::vector<int>& states,
        std::vector<int>& off_proc_states)
{
    int proc;
    int start, end;
    int ctr, prev_ctr, idx;
    int size, state;
    int n_sends = 0;
    int n_recvs = 0;
    int num_new_coarse = 0;
    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;
    std::vector<int> recv_indices;
    int tag = 4422;

    if (S->comm->send_data->size_msgs)
    {
        send_buffer.resize(S->comm->send_data->size_msgs);
    }
    if (S->comm->recv_data->size_msgs)
    {
        recv_buffer.resize(S->comm->recv_data->size_msgs);
        recv_indices.resize(S->comm->recv_data->size_msgs);
    }

    // Send states to procs in send_data
    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        proc = S->comm->send_data->procs[i];
        start = S->comm->send_data->indptr[i];
        end = S->comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->comm->send_data->indices[j];
            state = states[idx];
            if (state == -1 || state == 2)
            {
                send_buffer[ctr++] = state;
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Issend(&(send_buffer[prev_ctr]), size, MPI_INT, proc, tag,
                    MPI_COMM_WORLD, &(S->comm->send_data->requests[n_sends++]));
            prev_ctr = ctr;
        }
    }

    // Recv states of off_proc columns
    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
    {
        proc = S->comm->recv_data->procs[i];
        start = S->comm->recv_data->indptr[i];
        end = S->comm->recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            if (off_proc_states[j] == -1)
            {
                recv_indices[ctr++] = j;
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Irecv(&(recv_buffer[prev_ctr]), size, MPI_INT, proc, tag, 
                    MPI_COMM_WORLD, &(S->comm->recv_data->requests[n_recvs++]));
            prev_ctr = ctr;
        }
    }

    // Wait for communication to complete
    if (n_recvs)
    {
        MPI_Waitall(n_recvs, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
    }
    
    for (int i = 0; i < ctr; i++)
    {
        idx = recv_indices[i];
        off_proc_states[idx] = recv_buffer[i];
        if (off_proc_states[idx] == 2)
        {
            num_new_coarse++;
        }
    }

    if (n_sends)
    {
        MPI_Waitall(n_sends, S->comm->send_data->requests.data(), MPI_STATUSES_IGNORE);
    }

    return num_new_coarse;
}

void find_off_proc_new_coarse(const ParCSRMatrix* S,
        const std::map<int, int>& global_to_local,
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        std::vector<int>& off_proc_col_ptr,
        std::vector<int>& off_proc_col_coarse)
{
    int start, end;
    int idx, idx_k;
    int size;
    int idx_start, idx_end;
    int proc, count, buf_ptr;
    int ctr, prev_ctr;
    int tag = 2999;
    int global_col;
    int n_sends = 0;
    int msg_avail;
    MPI_Status recv_status;

    std::vector<int> send_ptr;
    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;

    // Clear out any previous data from off_proc_col_coarse
    //off_proc_col_coarse.clear();

    send_ptr.resize(S->comm->send_data->num_msgs + 1);

    // For each row in send_data, find if any new coarse
    // points in row.  If so, add to buffer.  Let buffer_ptr
    // point to the beginning/end of each send_data msg
    send_ptr[0] = 0;
    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        start = S->comm->send_data->indptr[i];
        end = S->comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->comm->send_data->indices[j];
            if (states[idx] == -1)
            {
                // Assuming num coarse in row is zero, 
                // but will update later
                buf_ptr = send_buffer.size();
                send_buffer.push_back(0);

                // Iterate through row
                idx_start = S->on_proc->idx1[idx];
                idx_end = S->on_proc->idx1[idx+1];
                if (S->on_proc->idx2[idx_start] == idx)
                {
                    idx_start++;
                }
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx_k = S->on_proc->idx2[k];
                    if (states[idx_k] == 2)
                    {
                        // New coarse, so add global idx to buffer
                        send_buffer.push_back(S->on_proc_column_map[idx_k]);
                    }
                }
                idx_start = S->off_proc->idx1[idx];
                idx_end = S->off_proc->idx1[idx+1];
                for (int k = idx_start; k < idx_end; k++)
                {
                    idx_k = S->off_proc->idx2[k];
                    if (off_proc_states[idx_k] == 2)
                    {
                        // New coarse, so add global idx to buffer
                        send_buffer.push_back(S->off_proc_column_map[idx_k]);
                    }
                }
                // Update buffer[ptr] with the number of coarse points
                // influenced by idx 
                send_buffer[buf_ptr] = send_buffer.size() - buf_ptr - 1;
            }
        }
        // Update buffer ptr (for send size)
        send_ptr[i+1] = send_buffer.size();
    }
 
    // Send coarse indices associated with each send idx
    // to appropriate procs
    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        proc = S->comm->send_data->procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        if (end - start)
        {
            MPI_Issend(&(send_buffer[start]), end - start, MPI_INT, proc,
                    tag, MPI_COMM_WORLD, &(S->comm->send_data->requests[n_sends++]));
        }
    }

    // Recv coarse indices influence by each off_proc column
    off_proc_col_ptr[0] = 0;
    for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
    {
        msg_avail = false;
        proc = S->comm->recv_data->procs[i];
        start = S->comm->recv_data->indptr[i];
        end = S->comm->recv_data->indptr[i+1];

        for (int j = start; j < end; j++)
        {
            if (off_proc_states[j] == -1)
            {
                msg_avail = true;
                break;
            }
        }
        if (msg_avail)
        {
            MPI_Probe(proc, tag, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_INT, &count);

            if (recv_buffer.size() < count)
            {
                recv_buffer.resize(count);
            }
            MPI_Recv(&recv_buffer[0], count, MPI_INT, proc, tag, MPI_COMM_WORLD,
                    &recv_status);
        }
        ctr = 0;
        for (int j = start; j < end; j++)
        {
            if (off_proc_states[j] == -1)
            {
                size = recv_buffer[ctr++];
                for (int k = 0; k < size; k++)
                {
                    global_col = recv_buffer[ctr++];
                    if (global_col >= S->partition->first_local_col 
                            && global_col <= S->partition->last_local_col)
                    {
                        off_proc_col_coarse.push_back(S->on_proc_partition_to_col[
                                global_col - S->partition->first_local_col]);
                    }
                    else
                    {
                        std::map<int, int>::const_iterator ptr = 
                        global_to_local.find(global_col);
                        if (ptr != global_to_local.end())
                        {
                            off_proc_col_coarse.push_back(ptr->second + S->on_proc_num_cols);
                        }   
                    }
                }
            }
            off_proc_col_ptr[j+1] = off_proc_col_coarse.size();
        }
    }

    // Wait for sends to complete
    if (n_sends)
    {
        MPI_Waitall(n_sends, S->comm->send_data->requests.data(), MPI_STATUSES_IGNORE);
    }
}

void combine_weight_updates(const ParCSRMatrix* S,
        const std::vector<int>&states,
        const std::vector<int>& off_proc_states,
        const std::vector<int>& off_proc_weight_updates,
        std::vector<double>& weights)
{
    int proc;
    int start, end;
    int ctr, prev_ctr;
    int idx, size;
    int n_sends = 0;
    int n_recvs = 0;
    int tag = 9233;

    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;
    std::vector<int> recv_indices;
    if (S->comm->recv_data->size_msgs)
    {
        send_buffer.resize(S->comm->recv_data->size_msgs);
    }
    if (S->comm->send_data->size_msgs)
    {
        recv_buffer.resize(S->comm->send_data->size_msgs);
        recv_indices.resize(S->comm->send_data->size_msgs);
    }

    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
    {
        proc = S->comm->recv_data->procs[i];
        start = S->comm->recv_data->indptr[i];
        end = S->comm->recv_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            if (off_proc_states[j] == -1)
            {
                send_buffer[ctr++] = off_proc_weight_updates[j];
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Issend(&(send_buffer[prev_ctr]), size, MPI_INT, proc, tag, 
                    MPI_COMM_WORLD, &(S->comm->recv_data->requests[n_sends++]));
            prev_ctr = ctr;
        }
    }

    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < S->comm->send_data->num_msgs; i++)
    {
        proc = S->comm->send_data->procs[i];
        start = S->comm->send_data->indptr[i];
        end = S->comm->send_data->indptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = S->comm->send_data->indices[j];
            if (states[idx] == -1)
            {
                recv_indices[ctr++] = idx;
            }
        }
        size = ctr - prev_ctr;
        if (size)
        {
            MPI_Irecv(&(recv_buffer[prev_ctr]), size, MPI_INT, proc, tag,
                    MPI_COMM_WORLD, &(S->comm->send_data->requests[n_recvs++]));
            prev_ctr = ctr;
        }
    }

    if (n_recvs)
    {
        MPI_Waitall(n_recvs, S->comm->send_data->requests.data(), MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < ctr; i++)
    {
        idx = recv_indices[i];
        weights[idx] += recv_buffer[i];
    }

    if (n_sends)
    {
        MPI_Waitall(n_sends, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
    }
}

int update_states(std::vector<double>& weights, 
        std::vector<int>& states)
{
    int num_states = states.size();
    int num_fine = 0;

    for (int i = 0; i < num_states; i++)
    {
        if (states[i] == -1)
        {
            if (weights[i] < 1)
            {
                states[i] = 0;
                num_fine++;
            }
        }
        else if (states[i] == 2)
        {
            states[i] = 1;
            weights[i] = 0;
        }
    }

    return num_fine;
}

void cljp_final_step(ParCSRMatrix* S,
        const std::map<int, int>& global_to_local,
        const std::vector<int>& states,
        int off_remaining,
        std::vector<int>& off_proc_col_ptr,
        std::vector<int>& off_proc_col_coarse,
        std::vector<int>& off_proc_weight_updates,
        std::vector<double>& off_proc_weights,
        std::vector<int>& off_proc_states,
        std::vector<int>& off_edgemark)
{
    int proc, start, end;
    int idx;
    int ctr, prev_ctr, size;
    int n_sends;
    int n_recvs;
    int off_num_fine;
    int off_num_coarse;
    int global_col;
    int count;
    int msg_avail;
    MPI_Status recv_status;
    std::vector<int> recv_buffer;
    std::vector<int> recv_indices;
    if (S->comm->recv_data->size_msgs)
    {
        recv_indices.resize(S->comm->recv_data->size_msgs);
        recv_buffer.resize(S->comm->recv_data->size_msgs);
    }

    while (off_remaining)
    {
        // Send max weight in col of each unassigned off_proc idx
        // Max weight is 0.0, as all local indices are labeled
        n_sends = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
        {
            proc = S->comm->recv_data->procs[i];
            start = S->comm->recv_data->indptr[i];
            end = S->comm->recv_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                if (off_proc_states[j] == -1)
                {
                    S->comm->recv_data->buffer[ctr++] = 0.0;
                }
            }
            size = ctr - prev_ctr;
            if (size)
            {
                MPI_Issend(&(S->comm->recv_data->buffer[prev_ctr]), size, 
                        MPI_DOUBLE, proc, 3266, MPI_COMM_WORLD, 
                        &(S->comm->recv_data->requests[n_sends++]));
                prev_ctr = ctr;
            }
        }
        if (n_sends)
        {
            MPI_Waitall(n_sends, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
        }

        // Recv new states for all unassigned off proc cols (new coarse)
        n_recvs = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
        {
            proc = S->comm->recv_data->procs[i];
            start = S->comm->recv_data->indptr[i];
            end = S->comm->recv_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                if (off_proc_states[j] == -1)
                {
                   recv_indices[ctr++] = j; 
                }
            }
            size = ctr - prev_ctr;
            if (size)
            {
                MPI_Irecv(&(recv_buffer[prev_ctr]), size, 
                        MPI_INT, proc, 4422, MPI_COMM_WORLD,
                        &(S->comm->recv_data->requests[n_recvs++]));
                prev_ctr = ctr;
            }
        }

        if (n_recvs)
        {
            MPI_Waitall(n_recvs, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
        }
        for (int i = 0; i < ctr; i++)
        {
            idx = recv_indices[i];
            off_proc_states[idx] = recv_buffer[i];
            if (recv_buffer[i] == 2)
            {
                off_remaining--;
            }
        }

        // Receive coarse indicies influenced by each unassigned off proc col
        off_proc_col_coarse.clear();

        off_proc_col_ptr[0] = 0;
        for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
        {
            msg_avail = false;
            proc = S->comm->recv_data->procs[i];
            start = S->comm->recv_data->indptr[i];
            end = S->comm->recv_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                if (off_proc_states[j] == -1)
                {
                    msg_avail = true;
                    break; 
                }
            }
            if (msg_avail)
            {
                MPI_Probe(proc, 2999, MPI_COMM_WORLD, &recv_status);
                MPI_Get_count(&recv_status, MPI_INT, &count);
        
                if (recv_buffer.size() < count)
                {
                    recv_buffer.resize(count);
                }
                MPI_Recv(&(recv_buffer[0]), count, MPI_INT, proc, 2999, MPI_COMM_WORLD,
                        &recv_status);
            }
            ctr = 0;
            for (int j = start; j < end; j++)
            {
                if (off_proc_states[j] == -1)
                {
                    size = recv_buffer[ctr++];
                    for (int k = 0; k < size; k++)
                    {
                        global_col = recv_buffer[ctr++];
                        if (global_col >= S->partition->first_local_col 
                                && global_col <= S->partition->last_local_col)
                        {
                            off_proc_col_coarse.push_back(S->on_proc_partition_to_col[
                                    global_col - S->partition->first_local_col]);
                        }
                        else
                        {
                            std::map<int, int>::const_iterator ptr = 
                                global_to_local.find(global_col);
                            if (ptr != global_to_local.end())
                            {
                                off_proc_col_coarse.push_back(ptr->second +
                                        S->on_proc_num_cols);
                            }
                        }
                    }
                }
                off_proc_col_ptr[j+1] = off_proc_col_coarse.size();
            }
        }

        // Update distance 2 off proc weights
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            off_proc_weight_updates[i] = 0;
        }
        // Update dist2 neighbors of new coarse
        update_weights_offproc_dist2(S, off_proc_col_ptr, off_proc_col_coarse,
                states, off_proc_states, off_proc_weight_updates, off_edgemark);

        // Send off proc weight updates
        n_sends = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
        {
            proc = S->comm->recv_data->procs[i];
            start = S->comm->recv_data->indptr[i];
            end = S->comm->recv_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                if (off_proc_states[j] == -1)
                {
                    recv_buffer[ctr++] = off_proc_weight_updates[j];
                }
            }
            size = ctr - prev_ctr;
            if (size)
            {
                MPI_Issend(&(recv_buffer[prev_ctr]), size, MPI_INT,
                        proc, 9233, MPI_COMM_WORLD, &(S->comm->recv_data->requests[n_sends++]));
                prev_ctr = ctr;
            }
        }
        if (n_sends)
        {
            MPI_Waitall(n_sends, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
        }

        // Find updates to off_proc_weights
        n_recvs = 0;
        ctr = 0;
        prev_ctr = 0;
        for (int i = 0; i < S->comm->recv_data->num_msgs; i++)
        {
            proc = S->comm->recv_data->procs[i];
            start = S->comm->recv_data->indptr[i];
            end = S->comm->recv_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                if (off_proc_states[j] == -1)
                {
                    recv_indices[ctr++] = j;
                }
                else
                {
                    off_proc_weights[j] = 0;
                }
            }
            size = ctr - prev_ctr;
            if (size)
            {
                MPI_Irecv(&(S->comm->recv_data->buffer[prev_ctr]), size, MPI_DOUBLE,
                        proc, 1992, MPI_COMM_WORLD, &(S->comm->recv_data->requests[n_recvs++]));
                prev_ctr = ctr;
            }
        }
        if (n_recvs)
        {
            MPI_Waitall(n_recvs, S->comm->recv_data->requests.data(), MPI_STATUSES_IGNORE);
        }

        for (int i = 0; i < ctr; i++)
        {
            idx = recv_indices[i];
            off_proc_weights[idx] = S->comm->recv_data->buffer[i];
        }

        // Update off proc states
        off_num_fine = update_states(off_proc_weights, off_proc_states);
        off_remaining -= off_num_fine;
    }
}

void cljp_main_loop(ParCSRMatrix* S,
        const std::vector<int>& on_col_ptr,
        const std::vector<int>& off_col_ptr,
        const std::vector<int>& on_col_indices,
        const std::vector<int>& off_col_indices,
        std::vector<double>& weights,
        std::vector<int>& states,
        std::vector<int>& off_proc_states,
        int remaining,
        std::vector<int>& on_edgemark,
        std::vector<int>& off_edgemark)
{
    /**********************************************
     * Declare and Initialize Variables
     **********************************************/
    int proc, idx;
    int start, end;
    int num_new_coarse;
    int off_num_new_coarse;
    int num_fine;
    int off_num_fine;
    int unassigned_off_proc;
    int off_remaining;

    std::vector<double> max_weights;
    std::vector<int> weight_updates;
    std::vector<double> off_proc_weights;
    std::vector<int> off_proc_col_coarse;
    std::vector<int> off_proc_weight_updates;
    std::vector<int> off_proc_col_ptr;
    std::map<int, int> global_to_local;

    if (S->local_num_rows)
    {
        max_weights.resize(S->local_num_rows);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_weight_updates.resize(S->off_proc_num_cols);
        off_proc_weights.resize(S->off_proc_num_cols, 0);
        off_proc_states.resize(S->off_proc_num_cols);
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            off_proc_states[i] = -1;
        }
    }
    if (S->comm->send_data->size_msgs)
    {
        weight_updates.resize(S->comm->send_data->size_msgs);
    }
    off_proc_col_ptr.resize(S->off_proc_num_cols + 1);

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        global_to_local[S->off_proc_column_map[i]] = i;
    }

    /**********************************************
     * Find weights of unassigned neighbors (off proc cols)
     **********************************************/
    find_off_proc_weights(S, states, off_proc_states, 
            weights, off_proc_weights);
    off_remaining = S->off_proc_num_cols;

    /**********************************************
     * While any local vertices still need assigned,
     * select independent set and update weights
     * accordingly (select new C/F points)
     **********************************************/
    while (remaining)
    {
        /**********************************************
        * For each local row i, find max weight in 
        * column i on all other processors (max_weights)
        **********************************************/
        find_max_off_weights(S, off_col_ptr, off_col_indices, 
                states, off_proc_states, weights, max_weights);

        /**********************************************
        * Select independent set: all indices with
        * maximum weight among unassigned neighbors
        **********************************************/
        num_new_coarse = select_independent_set(S, 
                weights, off_proc_weights, max_weights, on_col_ptr, 
                on_col_indices, states);
        remaining -= num_new_coarse;

        /**********************************************
        * Communicate updated states to neighbors
        * Only communicating previously unassigned states
        **********************************************/
        off_num_new_coarse = find_off_proc_states(S, states, off_proc_states);
        off_remaining -= off_num_new_coarse;

        /**********************************************
        * Find new coarse influenced by each off_proc col
        **********************************************/
        find_off_proc_new_coarse(S, global_to_local, states, off_proc_states, 
                off_proc_col_ptr, off_proc_col_coarse);

        /**********************************************
        * Update Weights
        **********************************************/
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            off_proc_weight_updates[i] = 0;
        }
        // Update local weights (for each local coarse row)
        update_weights(S, states, off_proc_states, weights, 
                off_proc_weight_updates, on_edgemark, off_edgemark);
        // Update dist2 neighbors of new coarse (on_proc)
        update_weights_onproc_dist2(S, on_col_ptr, off_col_ptr,
                on_col_indices, off_col_indices, states,
                off_proc_states, weights, on_edgemark);
        // Update dist2 neighbors of new coarse (off_proc)
        update_weights_offproc_dist2(S, off_proc_col_ptr, off_proc_col_coarse,
                states, off_proc_states, off_proc_weight_updates, off_edgemark);

        /**********************************************
        * Communicate off proc weight updates and
        * add recv'd updates to local weights
        **********************************************/
        combine_weight_updates(S, states, off_proc_states,
                off_proc_weight_updates, weights);

        /**********************************************
        * Find weights of unassigned neighbors (off proc cols)
        **********************************************/
        find_off_proc_weights(S, states, off_proc_states, 
                weights, off_proc_weights);

        /**********************************************
        * Update states, changing any new coarse states
        * from 2 to 1 (and changes weight to 0) and
        * set state of any unassigned indices with weight
        * less than 1 to fine (also update off_proc_states)
        **********************************************/
        num_fine = update_states(weights, states);
        off_num_fine = update_states(off_proc_weights,
                off_proc_states);
        remaining -= num_fine;
        off_remaining -= off_num_fine;
    }

    /**********************************************
    * While unassigned off_proc columns, need 
    * to keep communicating to neighbor
    **********************************************/
    if (off_remaining)
    {
        cljp_final_step(S, global_to_local, states, off_remaining,
                off_proc_col_ptr, off_proc_col_coarse, 
                off_proc_weight_updates, off_proc_weights, off_proc_states,
                off_edgemark);
    }
}

/**************************************************************
 *****   C/F Splitting
 **************************************************************
 *****  Assigns C (coarse) and F (fine) points to each local
 *****  index using Falgout coarsening:
 *****  Ruge-Stuben on interior nodes followed by CLJP on
 *****  all boundary nodes
 *****
 ***** Parameters
 ***** -------------
 ***** S : ParCSRMatrix*
 *****    Strength of connection matrix
 **************************************************************/
void split_rs(ParCSRMatrix* S,
        std::vector<int>& states, 
        std::vector<int>& off_proc_states)
{
    // Allocate space
    if (S->on_proc_num_cols)
    {
        states.resize(S->on_proc_num_cols);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_states.resize(S->off_proc_num_cols);
    }

    // Call serial ruge-stuben cf_splitting
    cf_splitting((CSRMatrix*) S->on_proc, states);

    if (S->comm == NULL)
    {
        S->comm = new ParComm(S->partition, S->off_proc_column_map, S->on_proc_column_map);
    }

    // Find states of off_proc_cols
    S->comm->communicate(states);
    std::copy(S->comm->recv_data->int_buffer.begin(), 
            S->comm->recv_data->int_buffer.end(), off_proc_states.begin());
}



/**************************************************************
 *****   C/F Splitting
 **************************************************************
 *****  Assigns C (coarse) and F (fine) points to each local
 *****  index using Falgout coarsening:
 *****  Ruge-Stuben on interior nodes followed by CLJP on
 *****  all boundary nodes
 *****
 ***** Parameters
 ***** -------------
 ***** S : ParCSRMatrix*
 *****    Strength of connection matrix
 **************************************************************/
void split_falgout(ParCSRMatrix* S,
        std::vector<int>& states, 
        std::vector<int>& off_proc_states)
{
    int start, end;
    int idx, idx_k;
    int idx_start, idx_end;
    int head, length, tmp;
    int remaining;
    int max_idx;
    double max_weight;

    std::vector<int> boundary;
    std::vector<int> on_col_ptr;
    std::vector<int> on_col_indices;
    std::vector<int> off_col_ptr;
    std::vector<int> off_col_indices;
    std::vector<int> on_edgemark;
    std::vector<int> off_edgemark;
    std::vector<double> weights;

    if (S->local_num_rows)
    {
        boundary.resize(S->local_num_rows, 0);
        weights.resize(S->local_num_rows, 0);
    }
    if (S->on_proc->nnz)
    {
        on_edgemark.resize(S->on_proc->nnz, 1);
    }
    if (S->off_proc->nnz)
    {
        off_edgemark.resize(S->off_proc->nnz, 1);
    }
        
    /**********************************************
     * Find which local rows are on boundary 
     * 1 - boundary
     * 0 - interior
     **********************************************/
    for (int i = 0; i < S->local_num_rows; i++)
    {
        // If cols in off_proc, i is boundary
        if (S->off_proc->idx1[i+1] - S->off_proc->idx1[i])
        {
            boundary[i] = 1;
        }
    }
    for (int i = 0; i < S->comm->send_data->size_msgs; i++)
    {
        // idx in send_data, so idx is boundary
        idx = S->comm->send_data->indices[i];
        boundary[idx] = 1;
    }

    /**********************************************
     * Form S^T ptr and indices for on_proc and off_proc
     * Needed for column-wise searches (which indices
     * influence each on_proc/off_proc index)
     **********************************************/
    binary_transpose(S, on_col_ptr, off_col_ptr,
            on_col_indices, off_col_indices);

    /**********************************************
     * Ruge-Stuben on local portion
     **********************************************/
    cf_splitting((CSRMatrix*)S->on_proc, states);

    /**********************************************
     * Reset states of boundary indices
     **********************************************/
    remaining = 0;
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (boundary[i])
        {
            states[i] = -1;
            remaining++;
        }
    }

    /**********************************************
     * Set initial CLJP boundary weights
     **********************************************/
    initial_cljp_weights(S, on_col_ptr, on_col_indices,
            states, weights, on_edgemark);

    /**********************************************
     * CLJP Main Loop
     **********************************************/
    cljp_main_loop(S, on_col_ptr, off_col_ptr,
            on_col_indices, off_col_indices, weights,
            states, off_proc_states, remaining, on_edgemark,
            off_edgemark);
}

/**************************************************************
 *****   C/F Splitting
 **************************************************************
 *****  Assigns C (coarse) and F (fine) points to each local
 *****  index using Falgout coarsening:
 *****  Ruge-Stuben on interior nodes followed by CLJP on
 *****  all boundary nodes
 *****
 ***** Parameters
 ***** -------------
 ***** S : ParCSRMatrix*
 *****    Strength of connection matrix
 **************************************************************/
void split_cljp(ParCSRMatrix* S,
        std::vector<int>& states, 
        std::vector<int>& off_proc_states)
{
    int start, end;
    int idx, idx_k;
    int idx_start, idx_end;
    int head, length, tmp;
    int remaining;
    int max_idx;
    double max_weight;

    std::vector<int> on_col_ptr;
    std::vector<int> on_col_indices;
    std::vector<int> off_col_ptr;
    std::vector<int> off_col_indices;
    std::vector<int> on_edgemark;
    std::vector<int> off_edgemark;
    std::vector<double> weights;

    if (S->local_num_rows)
    {
        weights.resize(S->local_num_rows, 0);
    }
    if (S->on_proc->nnz)
    {
        on_edgemark.resize(S->on_proc->nnz, 1);
    }
    if (S->off_proc->nnz)
    {
        off_edgemark.resize(S->off_proc->nnz, 1);
    }
        
    /**********************************************
     * Form S^T ptr and indices for on_proc and off_proc
     * Needed for column-wise searches (which indices
     * influence each on_proc/off_proc index)
     **********************************************/
    binary_transpose(S, on_col_ptr, off_col_ptr,
            on_col_indices, off_col_indices);

    /**********************************************
     * Reset states of boundary indices
     **********************************************/
    if (S->local_num_rows)
    {
        states.resize(S->local_num_rows);
    }
    remaining = S->local_num_rows;
    for (int i = 0; i < S->local_num_rows; i++)
    {
        states[i] = -1;
    }

    /**********************************************
     * Set initial CLJP boundary weights
     **********************************************/
    initial_cljp_weights(S, on_col_ptr, on_col_indices,
            states, weights, on_edgemark);

    /**********************************************
     * CLJP Main Loop
     **********************************************/
    cljp_main_loop(S, on_col_ptr, off_col_ptr,
            on_col_indices, off_col_indices, weights,
            states, off_proc_states, remaining, on_edgemark,
            off_edgemark);
}




