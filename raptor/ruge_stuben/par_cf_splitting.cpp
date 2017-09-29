// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_cf_splitting.hpp"

using namespace raptor;

void transpose(const ParCSRMatrix* S,
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
        std::vector<double>& weights)
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

    if (S->off_proc_num_cols)
    {
        off_proc_weights.resize(S->off_proc_num_cols, 0);
    }
    if (S->comm->send_data->size_msgs)
    {
        recv_weights.resize(S->comm->send_data->size_msgs);
    }

    FILE* f = fopen("../../tests/weights.txt", "r");
    // Set each weight initially to random value [0,1)
    int first_row = 0;
    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<int> proc_sizes(num_procs);
    MPI_Allgather(&S->on_proc_num_cols, 1, MPI_INT, proc_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }
    for (int i = 0; i < first_row; i++)
    {
        double weight;
        fscanf(f, "%lg\n", &weight);
    }
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        // Seed random generator with global col
        //srand(S->on_proc_column_map[i]);

        // Random value [0,1)
        //weights[i] = rand();
        //if (weights[i] == RAND_MAX)
        //    weights[i] -= 1;
        //weights[i] /= RAND_MAX;
        fscanf(f, "%lg\n", &weights[i]);
    }
    fclose(f);

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
        const int remaining,
        const std::vector<int>& unassigned,
        const std::vector<double>& weights, 
        const std::vector<double>& off_proc_weights,
        const std::vector<double>& max_off_weights,
        const std::vector<int>& on_col_ptr,
        const std::vector<int>& on_col_indices,
        std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        std::vector<int>& new_coarse_list)
{
    int start, end, idx;
    int col;
    int j, u;
    int num_new_coarse = 0;
    double weight;

    for (int i = 0; i < remaining; i++)
    {
        u = unassigned[i];
        weight = weights[u];

        // Compare to max weight (with unassigned state), strongly connected
        // in a column of S (off_proc)
        if (max_off_weights[u] > weight)
        {
           continue; 
        }

        // Compare to weights of unassigned states, strongly connected in
        // row of S (on_proc)
        start = S->on_proc->idx1[u];
        end = S->on_proc->idx1[u+1];
        if (S->on_proc->idx2[start] == u)
        {
            start++;
        }
        for (j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (weights[idx] > weight)
            {
                break;
            }
        }
        if (j != end)
        {
            continue;
        }

        // Compare to weights of unassigned states, strongly connected in
        // row of S (off_proc)
        start = S->off_proc->idx1[u];
        end = S->off_proc->idx1[u+1];
        for (j = start; j < end; j++)
        {
            idx = S->off_proc->idx2[j];
            if (off_proc_weights[idx] > weight)
            {
                break;
            }
        }
        if (j != end)
        {
            continue;
        }
               
        // Compare to weights of unassigned states, strongly connected in
        // column of S (on_proc)
        start = on_col_ptr[u];
        end = on_col_ptr[u+1];
        for (j = start; j < end; j++)
        {
            idx = on_col_indices[j];
            if (weights[idx] > weight)
            {
                break;
            }
        }
        if (j != end)
        {
            continue;
        }

        // If i made it this far, weight is greater than all unassigned
        // neighbors
        states[u] = 2;
        new_coarse_list[num_new_coarse++] = u;
    }

    return num_new_coarse;
}

void update_row_weights(const ParCSRMatrix* S,
        const int num_new_coarse,
        const std::vector<int>& new_coarse_list,
        std::vector<int>& on_edgemark, 
        std::vector<int>& off_edgemark,
        const std::vector<int>& states,
        const std::vector<int>& off_proc_states, 
        std::vector<double>& weights,
        std::vector<int>& off_proc_weight_updates)
{
    int start, end;
    int c, idx;

    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];

        start = S->on_proc->idx1[c];
        end = S->on_proc->idx1[c+1];
        if (S->on_proc->idx2[start] == c)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (states[idx] == -1 && on_edgemark[j])
            {
                on_edgemark[j] = 0;
                weights[idx]--;
            }
        }

        start = S->off_proc->idx1[c];
        end = S->off_proc->idx1[c+1];
        for (int j = start; j < end; j++)
        {
            idx = S->off_proc->idx2[j];
            if (off_proc_states[idx] == -1 && off_edgemark[j])
            {
                off_edgemark[j] = 0;
                off_proc_weight_updates[idx]--;
            }
        }
    }
}

void update_local_dist2_weights(const ParCSRMatrix* S,
        const int num_new_coarse,
        const std::vector<int>& new_coarse_list,
        const int off_num_new_coarse, 
        const std::vector<int>& off_new_coarse_list,
        const std::vector<int>& on_col_ptr,
        const std::vector<int>& on_col_indices,
        const std::vector<int>& off_col_ptr,
        const std::vector<int>& off_col_indices,
        std::vector<int>& on_edgemark, 
        const std::vector<int>& states, 
        std::vector<double>& weights)
{
    int start, end;
    int c, idx, idx_k;
    int idx_start, idx_end;

    std::vector<int> c_dep_cache;
    if (S->on_proc_num_cols)
    {
        c_dep_cache.resize(S->on_proc_num_cols, -1);
    }

    // Update local weights based on new_coarse values
    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];

        // Find rows strongly connected to c and add to c_dep_cache
        start = on_col_ptr[c];
        end = on_col_ptr[c+1];
        for (int j = start; j < end; j++)
        {
            idx = on_col_indices[j];
            if (states[idx] == -1)
            {
                c_dep_cache[idx] = c;
            }
        }

        // Go through each row i strongly connected to c, and see if any column
        // in Si is also in c_dep_cache (and edge not already removed), reduce
        // weight and remove edge
        for (int j = start; j < end; j++)
        {
            idx = on_col_indices[j];
            if (states[idx] == 1) continue;

            idx_start = S->on_proc->idx1[idx];
            idx_end = S->on_proc->idx1[idx+1];
            if (S->on_proc->idx2[idx_start] == idx)
            {
                idx_start++;
            }
            for (int k = idx_start; k < idx_end; k++)
            {
                idx_k = S->on_proc->idx2[k];
                if (states[idx_k] == -1 && on_edgemark[k] && c_dep_cache[idx_k] == c)
                {
                    on_edgemark[k] = 0;
                    weights[idx_k]--;
                }
            }
        }
    }

    // Update local weights based on off_proc new coarse values
    for (int i = 0; i < off_num_new_coarse; i++)
    {
        c = off_new_coarse_list[i];

        // Find rows strongly connected to c and add to c_dep_cache
        start = off_col_ptr[c];
        end = off_col_ptr[c+1];
        for (int j = start; j < end; j++)
        {
            idx = off_col_indices[j];
            if (states[idx] == -1)
            {
                c_dep_cache[idx] = c;
            }
        }

        // Go through each row i strongly conected to c, and see if any column
        // in Si is also in c_dep_cache (and edge not already removed), reduce
        // weight and remove edge
        for (int j = start; j < end; j++)
        {
            idx = off_col_indices[j];
            if (states[idx] == 1) continue;

            idx_start = S->on_proc->idx1[idx];
            idx_end = S->on_proc->idx1[idx+1];
            if (S->on_proc->idx2[idx_start] == idx)
            {
                idx_start++;
            }
            for (int k = idx_start; k < idx_end; k++)
            {
                idx_k = S->on_proc->idx2[k];
                if (states[idx_k] == -1 && on_edgemark[k] && c_dep_cache[idx_k] == c)
                {
                    on_edgemark[k] = 0;
                    weights[idx_k]--;
                }
            }
        }
    }
}

void update_off_proc_dist2_weights(const ParCSRMatrix* S,
        const int num_new_coarse,
        const int off_num_new_coarse,
        const std::vector<int> new_coarse_list,
        const std::vector<int> off_new_coarse_list,
        const std::vector<int>& recv_off_col_ptr,
        const std::vector<int>& recv_off_col_coarse,
        const std::vector<int>& on_col_ptr,
        const std::vector<int>& on_col_indices,
        const std::vector<int>& off_col_ptr,
        const std::vector<int>& off_col_indices,
        std::vector<int>& off_edgemark, 
        const std::vector<int>& off_proc_states, 
        std::vector<int>& off_proc_weight_updates)
{
    int start, end;
    int c, idx, idx_k;
    int on_nnz, off_nnz;
    int idx_start, idx_end;
    int new_idx;

    std::vector<int> on_sizes;
    std::vector<int> on_ptr(num_new_coarse+1);
    std::vector<int> on_indices;
    std::vector<int> off_sizes;
    std::vector<int> off_ptr(off_num_new_coarse+1);
    std::vector<int> off_indices;
    std::vector<int> on_proc_col_to_coarse;
    std::vector<int> off_proc_col_to_coarse;
    std::map<int, int> map_to_local;
    
    std::vector<int> c_dep_cache;
    if (S->off_proc_num_cols)
    {
        c_dep_cache.resize(S->off_proc_num_cols, -1);
    }

    // Map global off_proc columns to local indices
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        map_to_local[S->off_proc_column_map[i]] = i;
    }

    // Map index i in on(/off)_proc_num_cols to coarse_list
    if (S->on_proc_num_cols)
    {
        on_proc_col_to_coarse.resize(S->on_proc_num_cols);
        for (int i = 0; i < num_new_coarse; i++)
        {
            on_proc_col_to_coarse[new_coarse_list[i]] = i;
        }
    }
    if (S->off_proc_num_cols)
    {
        off_proc_col_to_coarse.resize(S->off_proc_num_cols);
        for (int i = 0; i < off_num_new_coarse; i++)
        {
            off_proc_col_to_coarse[off_new_coarse_list[i]] = i;
        }
    }

    // Find the number of off_proc cols recving each new coarse / off new coarse
    on_nnz = 0;
    off_nnz = 0;
    if (num_new_coarse)
    {
        on_sizes.resize(num_new_coarse, 0);
    }
    if (off_num_new_coarse)
    {
        off_sizes.resize(off_num_new_coarse, 0);
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        start = recv_off_col_ptr[i];
        end = recv_off_col_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = recv_off_col_coarse[j];
            if (idx < S->on_proc_num_cols)
            {
                on_sizes[on_proc_col_to_coarse[idx]]++;
                on_nnz++;
            }
            else
            {
                idx -= S->on_proc_num_cols;
                off_sizes[off_proc_col_to_coarse[idx]]++;
                off_nnz++;
            }
        }
    }

    // Create ptrs for off_proc cols associated with new coarse
    on_ptr[0] = 0;
    for (int i = 0; i < num_new_coarse; i++)
    {
        on_ptr[i+1] = on_ptr[i] + on_sizes[i];
        on_sizes[i] = 0;
    }
    off_ptr[0] = 0;
    for (int i = 0; i < off_num_new_coarse; i++)
    {
        off_ptr[i+1] = off_ptr[i] + off_sizes[i];
        off_sizes[i] = 0;
    }

    // Add off_proc_cols to on_indices and off_indices (ordered by associated
    // coarse column)
    if (on_nnz)
    {
        on_indices.resize(on_nnz);
    }
    if (off_nnz)
    {
        off_indices.resize(off_nnz);
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        start = recv_off_col_ptr[i];
        end = recv_off_col_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = recv_off_col_coarse[j];
            if (idx < S->on_proc_num_cols)
            {
                idx = on_proc_col_to_coarse[idx];
                new_idx = on_ptr[idx] + on_sizes[idx]++;
                on_indices[new_idx] = i;            }
            else
            {
                idx = off_proc_col_to_coarse[idx - S->on_proc_num_cols];
                new_idx = off_ptr[idx] + off_sizes[idx]++;
                off_indices[new_idx] = i;
            }
        }
    }

    // Go through on_proc coarse columns 
    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];

        // Add recvd values to c_dep_cache
        start = on_ptr[i];
        end = on_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = on_indices[j];
            if (off_proc_states[idx] == -1)
            {
                c_dep_cache[idx] = c;
            }
        }

        // Iterate over local portion of coarse col c
        // For each row i such that Sic, iterate over row i
        // Find all cols j such that Sij = 1 (edgemark), and Sic (c_dep_cache)
        start = on_col_ptr[c];
        end = on_col_ptr[c+1];
        for (int j = start; j < end; j++)
        {
            idx = on_col_indices[j];
            idx_start = S->off_proc->idx1[idx];
            idx_end = S->off_proc->idx1[idx+1];
            for (int k = idx_start; k < idx_end; k++)
            {
                idx_k = S->off_proc->idx2[k];
                if (off_proc_states[idx_k] == -1 && off_edgemark[k] 
                        && c_dep_cache[idx_k] == c)
                {
                    off_edgemark[k] = 0;
                    off_proc_weight_updates[idx_k]--;
                }
            }
        }
    }

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        c_dep_cache[i] = -1;
    }

    // Go through off_proc coarse columns
    for (int i = 0; i < off_num_new_coarse; i++)
    {
        c = off_new_coarse_list[i];

        // Add recvd values to c_dep_cache
        start = off_ptr[i];
        end = off_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            idx = off_indices[j];
            if (off_proc_states[idx] == -1)
            {
                c_dep_cache[idx] = c;
            }
        }

        // Iterate over local portion of coarse col c
        // For each row i such that Sic, iterate over row i
        // Find all cols j such that Sij = 1 (edgemark) and Sic (c_dep_cache)
        start = off_col_ptr[c];
        end = off_col_ptr[c+1];
        for (int j = start; j < end; j++)
        {
            idx = off_col_indices[j];
            idx_start = S->off_proc->idx1[idx];
            idx_end = S->off_proc->idx1[idx+1];
            for (int k = idx_start; k < idx_end; k++)
            {
                idx_k = S->off_proc->idx2[k];
                if (off_proc_states[idx_k] == -1 && off_edgemark[k]
                        && c_dep_cache[idx_k] == c)
                {
                    off_edgemark[k] = 0;
                    off_proc_weight_updates[idx_k]--;
                }
            }
        }
    }
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

    off_proc_col_coarse.clear();

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
        std::vector<int>& states, const int remaining, std::vector<int>& unassigned)
{
    int num_states = states.size();
    int num_fine = 0;

    int ctr = 0;
    int u;
    for (int i = 0; i < remaining; i++)
    {
        u = unassigned[i];
        if (states[u] == 2)
        {
            weights[u] = 0.0;
            states[u] = 1;
        }
        else if (weights[u] < 1.0)
        {
            weights[u] = 0.0;
            states[u] = 0;
            num_fine++;
        }
        else
        {
            unassigned[ctr++] = unassigned[i];
        }
    }

    return ctr;
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
    int proc, idx, ctr;
    int start, end;
    int num_new_coarse;
    int off_num_new_coarse;
    int num_fine;
    int off_num_fine;
    int unassigned_off_proc;
    int off_remaining;
    int size, global_col;

    std::vector<double> max_weights;
    std::vector<int> weight_updates;
    std::vector<double> off_proc_weights;
    std::vector<int> off_proc_col_coarse;
    std::vector<int> off_proc_weight_updates;
    std::vector<int> off_proc_col_ptr;
    std::map<int, int> global_to_local;
    std::vector<int> new_coarse_list;
    std::vector<int> off_new_coarse_list;
    std::vector<int> unassigned;
    std::vector<int> unassigned_off;

    int count;
    int n_sends, n_recvs;
    int prev_ctr;
    int msg_avail;
    MPI_Status recv_status;
    std::vector<int> recv_buffer;
    std::vector<int> recv_indices;
    if (S->comm->recv_data->size_msgs)
    {
        recv_indices.resize(S->comm->recv_data->size_msgs);
        recv_buffer.resize(S->comm->recv_data->size_msgs);
    }

    if (S->local_num_rows)
    {
        max_weights.resize(S->local_num_rows);
        new_coarse_list.resize(S->local_num_rows);
        unassigned.resize(S->local_num_rows);
        std::iota(unassigned.begin(), unassigned.end(), 0);
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
        off_new_coarse_list.resize(S->off_proc_num_cols);
        unassigned_off.resize(S->off_proc_num_cols);
        std::iota(unassigned_off.begin(), unassigned_off.end(), 0);
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

    remaining = 0;
    off_remaining = 0;
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == -1)
        {
            unassigned[remaining++] = i;
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == -1)
        {
            unassigned_off[off_remaining++] = i;
        }
    }

    /**********************************************
     * While any local vertices still need assigned,
     * select independent set and update weights
     * accordingly (select new C/F points)
     **********************************************/
    while (remaining || off_remaining)
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
        num_new_coarse = select_independent_set(S, remaining, unassigned,
                weights, off_proc_weights, max_weights, on_col_ptr, 
                on_col_indices, states, off_proc_states, new_coarse_list);
        //remaining -= num_new_coarse;

        // Communicate updated states to neighbors
        // Only communicating previously unassigned states
        off_num_new_coarse = find_off_proc_states(S, states, off_proc_states);
        //off_remaining -= off_num_new_coarse;

        ctr = 0;
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            if (off_proc_states[i] == 2)
            {
                off_new_coarse_list[ctr++] = i;
            }
        }

        // Find new coarse influenced by each off_proc col
        find_off_proc_new_coarse(S, global_to_local, states, off_proc_states, 
                off_proc_col_ptr, off_proc_col_coarse);

        // Update Weights
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            off_proc_weight_updates[i] = 0;
        }
        update_row_weights(S, num_new_coarse, new_coarse_list, on_edgemark,
                off_edgemark, states, off_proc_states, weights, 
                off_proc_weight_updates);
        update_local_dist2_weights(S, num_new_coarse, new_coarse_list, 
                off_num_new_coarse, off_new_coarse_list, on_col_ptr, on_col_indices,
                off_col_ptr, off_col_indices, on_edgemark, states, weights);
        update_off_proc_dist2_weights(S, num_new_coarse, off_num_new_coarse, 
                new_coarse_list, off_new_coarse_list, off_proc_col_ptr, 
                off_proc_col_coarse, on_col_ptr, on_col_indices, off_col_ptr,
                off_col_indices, off_edgemark, off_proc_states, 
                off_proc_weight_updates);

        // Communicate off proc weight updates and
        // add recv'd updates to local weights
        combine_weight_updates(S, states, off_proc_states,
                off_proc_weight_updates, weights);

        // Find weights of unassigned neighbors (off proc cols)
        find_off_proc_weights(S, states, off_proc_states, 
                weights, off_proc_weights);

        // Update states, changing any new coarse states
        // from 2 to 1 (and changes weight to 0) and
        // set state of any unassigned indices with weight
        // less than 1 to fine (also update off_proc_states)
        remaining = update_states(weights, states, remaining, unassigned);
        off_remaining = update_states(off_proc_weights,
                off_proc_states, off_remaining, unassigned_off);
        //remaining -= num_fine;
        //off_remaining -= off_num_fine;
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
    split_rs((CSRMatrix*) S->on_proc, states);

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
    int idx, idx_k, c;
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
    transpose(S, on_col_ptr, off_col_ptr,
            on_col_indices, off_col_indices);

    /**********************************************
     * Ruge-Stuben on local portion
     **********************************************/
    split_rs((CSRMatrix*)S->on_proc, states);

    /**********************************************
     * Reset states of boundary indices
     **********************************************/
    remaining = 0;
    int num_new_coarse = 0;
    std::vector<int> new_coarse_list;
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (boundary[i])
        {
            states[i] = -1;
            remaining++;
        } 
        else if (states[i] == 1)
        {
            new_coarse_list.push_back(i);
        }
    }
    num_new_coarse = new_coarse_list.size();

    /**********************************************
     * Set initial CLJP boundary weights
     **********************************************/
    initial_cljp_weights(S, weights);

    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] != -1)
        {
            weights[i] = 0.0;
        }
    }

    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];

        start = S->on_proc->idx1[c];
        end = S->on_proc->idx1[c+1];
        if (S->on_proc->idx2[start] == c)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->on_proc->idx2[j];
            if (states[idx] == -1 && on_edgemark[j])
            {
                on_edgemark[j] = 0;
                weights[idx]--;
            }
        }
    }

    std::vector<int> c_dep_cache;
    if (S->on_proc_num_cols)
    {
        c_dep_cache.resize(S->on_proc_num_cols, -1);
    }

    // Update local weights based on new_coarse values
    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];

        // Find rows strongly connected to c and add to c_dep_cache
        start = on_col_ptr[c];
        end = on_col_ptr[c+1];
        for (int j = start; j < end; j++)
        {
            idx = on_col_indices[j];
            if (states[idx] == -1)
            {
                c_dep_cache[idx] = c;
            }
        }

        // Go through each row i strongly connected to c, and see if any column
        // in Si is also in c_dep_cache (and edge not already removed), reduce
        // weight and remove edge
        for (int j = start; j < end; j++)
        {
            idx = on_col_indices[j];

            idx_start = S->on_proc->idx1[idx];
            idx_end = S->on_proc->idx1[idx+1];
            if (S->on_proc->idx2[idx_start] == idx)
            {
                idx_start++;
            }
            for (int k = idx_start; k < idx_end; k++)
            {
                idx_k = S->on_proc->idx2[k];
                if (states[idx_k] == -1 && on_edgemark[k] && c_dep_cache[idx_k] == c)
                {
                    on_edgemark[k] = 0;
                    weights[idx_k]--;
                }
            }
        }
    }


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
    int remaining;
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
    transpose(S, on_col_ptr, off_col_ptr,
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
    initial_cljp_weights(S, weights);

    /**********************************************
     * CLJP Main Loop
     **********************************************/
    cljp_main_loop(S, on_col_ptr, off_col_ptr,
            on_col_indices, off_col_indices, weights,
            states, off_proc_states, remaining, on_edgemark,
            off_edgemark);

}




