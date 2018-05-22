// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_cf_splitting.hpp"

using namespace raptor;

void split_rs(ParCSRMatrix* S, aligned_vector<int>& states, 
        aligned_vector<int>& off_proc_states, bool tap_cf, data_t* comm_t)
{
    CommPkg* comm = S->comm;
    if (tap_cf)
    {
        comm = S->tap_comm;
    }

    set_initial_states(S, states);

    // Call serial ruge-stuben cf_splitting
    split_rs((CSRMatrix*) S->on_proc, states, true);

    // Find states of off_proc_cols
    if (S->off_proc_num_cols)
    {
        off_proc_states.resize(S->off_proc_num_cols);
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<int>& recvbuf = comm->communicate(states);
    if (comm_t) *comm_t += MPI_Wtime();

    std::copy(recvbuf.begin(), recvbuf.end(), off_proc_states.begin());
}

void split_cljp(ParCSRMatrix* S, aligned_vector<int>& states, 
        aligned_vector<int>& off_proc_states, bool tap_cf, 
        double* rand_vals, data_t* comm_t, data_t* comm_mat_t)
{
    
    S->on_proc->move_diag();

    /**********************************************
     * Reset states of boundary indices
     **********************************************/
    set_initial_states(S, states);

    /**********************************************
     * CLJP Main Loop
     **********************************************/
    cljp_main_loop(S, states, off_proc_states, tap_cf, 
            rand_vals, comm_t, comm_mat_t);
}

void split_falgout(ParCSRMatrix* S, aligned_vector<int>& states, 
        aligned_vector<int>& off_proc_states, bool tap_cf, 
        double* rand_vals, data_t* comm_t, data_t* comm_mat_t)
{
    S->on_proc->move_diag();

    /**********************************************
     * Reset states of boundary indices
     **********************************************/
    set_initial_states(S, states);

    /**********************************************
     * Ruge-Stuben on local portion
     **********************************************/
    split_rs((CSRMatrix*)S->on_proc, states, true);

    reset_boundaries(S, states);

    /**********************************************
     * CLJP Main Loop
     **********************************************/
    cljp_main_loop(S, states, off_proc_states, tap_cf, 
            rand_vals, comm_t, comm_mat_t);
}

void split_pmis(ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, bool tap_cf, 
        double* rand_vals, data_t* comm_t)
{
    int remaining;

    S->on_proc->move_diag();

    set_initial_states(S, states);

    /**********************************************
     * CLJP Main Loop
     **********************************************/
    pmis_main_loop(S, states, off_proc_states, tap_cf, rand_vals, comm_t);
}

void split_hmis(ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, bool tap_cf, 
        double* rand_vals, data_t* comm_t)
{
    S->on_proc->move_diag();

    set_initial_states(S, states);

    /**********************************************
     * Ruge-Stuben on local portion
     * Initialized States = true
     * Second pass of RS = false
     **********************************************/
    split_rs((CSRMatrix*)S->on_proc, states, true, false); // Only first pass

    reset_boundaries(S, states);

    /**********************************************
     * PMIS Main Loop
     **********************************************/
    pmis_main_loop(S, states, off_proc_states, tap_cf, rand_vals, comm_t);
}

void set_initial_states(ParCSRMatrix* S, aligned_vector<int>& states)
{
    if (S->local_num_rows == 0) return;

    states.resize(S->local_num_rows);
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (S->on_proc->idx1[i+1] - S->on_proc->idx1[i] > 1
                || S->off_proc->idx1[i+1] - S->off_proc->idx1[i])
        {
            states[i] = -1;
        }
        else
        {
            states[i] = -3;
        }
    }
}

void reset_boundaries(ParCSRMatrix* S, aligned_vector<int>& states)
{
    if (S->local_num_rows == 0) return;

    aligned_vector<int> boundary(S->local_num_rows, 0);
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (S->off_proc->idx1[i+1] - S->off_proc->idx1[i])
        {
            boundary[i] = 1;
        }
    }
    for (int i = 0; i < S->comm->send_data->size_msgs; i++)
    {
        boundary[S->comm->send_data->indices[i]] = 1;
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (boundary[i])
        {
            states[i] = -1;
        }
    }
}

void transpose(const ParCSRMatrix* S,
        aligned_vector<int>& on_col_ptr, 
        aligned_vector<int>& off_col_ptr, 
        aligned_vector<int>& on_col_indices,
        aligned_vector<int>& off_col_indices)
{
    int start, end;
    int col, idx;
    aligned_vector<int> on_col_sizes;
    aligned_vector<int> off_col_sizes;

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

void initial_weights(const ParCSRMatrix* S,
        CommPkg* comm, 
        aligned_vector<double>& weights, 
        double* rand_vals = NULL)
{
    int start, end;
    int idx;
    
    
    int proc;
    int tag = 3029;
    

    aligned_vector<int> off_proc_weights;

    if (S->off_proc_num_cols)
    {
        off_proc_weights.resize(S->off_proc_num_cols, 0);
    }

    // Set each weight initially to random value [0,1)
    if (rand_vals)
    {
        for (int i = 0; i < S->on_proc_num_cols; i++)
        {
            weights[i] = rand_vals[i];
        }
    }
    else
    {
        srand(time(NULL));
        for (int i = 0; i < S->on_proc_num_cols; i++)
        {
            weights[i] = ((double)(rand())) / RAND_MAX;
        }
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

    comm->communicate_T(off_proc_weights, weights);
}

void find_off_proc_weights(CommPkg* comm, 
        const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        const aligned_vector<double>& weights,
        aligned_vector<double>& off_proc_weights,
        bool first_pass = false)
{
    int off_proc_num_cols = off_proc_states.size();

    if (first_pass)
    {
        aligned_vector<double>& recvbuf = comm->communicate(weights);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            off_proc_weights[i] = recvbuf[i];
        }
    }
    else
    {
        std::function<bool(int)> compare_func = [](const int a)
        {
            return a == -1;
        };
        aligned_vector<double>& recvbuf = ((ParComm*)comm)->conditional_comm(weights, states,
                off_proc_states, compare_func);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            off_proc_weights[i] = recvbuf[i];
        }
    }
}

void find_max_off_weights(CommPkg* comm,
        const aligned_vector<int>& off_col_ptr,
        const aligned_vector<int>& off_col_indices,
        const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        const aligned_vector<double>& weights,
        aligned_vector<double>& max_weights,
        bool first_pass = false)
{
    int start, end, idx;
    double max_weight;

    int off_proc_num_cols = off_proc_states.size();

    aligned_vector<double> send_weights;
    if (off_proc_num_cols)
    {
        send_weights.resize(off_proc_num_cols);
    }
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == -1 || first_pass)
        {
            max_weight = 0;
            start = off_col_ptr[i];
            end = off_col_ptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = off_col_indices[j];
                if (weights[idx] > max_weight)
                {
                    max_weight = weights[idx];
                }
            }
            send_weights[i] = max_weight;
        }
    }
    std::fill(max_weights.begin(), max_weights.end(), 0);
    std::function<double(double, double)> result_max = [](double c, double d)
    {
        if (c > d) return c;
        else return d;
    };

    if (first_pass)
    {
        comm->communicate_T(send_weights, max_weights, result_max);
    }
    else
    {
        std::function<bool(int)> compare_func = [](const int a)
        {
            return a == -1;
        };
        ((ParComm*)comm)->conditional_comm_T(send_weights, states, off_proc_states, 
                compare_func, max_weights, result_max);
    }
}

int select_independent_set(const ParCSRMatrix* S, 
        const int remaining,
        const aligned_vector<int>& unassigned,
        const aligned_vector<double>& weights, 
        const aligned_vector<double>& off_proc_weights,
        const aligned_vector<double>& max_off_weights,
        const aligned_vector<int>& on_col_ptr,
        const aligned_vector<int>& on_col_indices,
        aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        aligned_vector<int>& new_coarse_list)
{
    int start, end, idx;
    
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
        const aligned_vector<int>& new_coarse_list,
        aligned_vector<int>& on_edgemark, 
        aligned_vector<int>& off_edgemark,
        const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states, 
        aligned_vector<double>& weights,
        aligned_vector<int>& off_proc_weight_updates)
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
        const aligned_vector<int>& new_coarse_list,
        const int off_num_new_coarse, 
        const aligned_vector<int>& off_new_coarse_list,
        const aligned_vector<int>& on_col_ptr,
        const aligned_vector<int>& on_col_indices,
        const aligned_vector<int>& off_col_ptr,
        const aligned_vector<int>& off_col_indices,
        aligned_vector<int>& on_edgemark, 
        const aligned_vector<int>& states, 
        aligned_vector<double>& weights)
{
    int start, end;
    int c, idx, idx_k;
    int idx_start, idx_end;

    aligned_vector<int> c_dep_cache;
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

    if (off_num_new_coarse)
    {
        for (int i = 0; i < S->local_num_rows; i++)
        {
            c_dep_cache[i] = -1;
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
        const aligned_vector<int> new_coarse_list,
        const aligned_vector<int> off_new_coarse_list,
        const aligned_vector<int>& recv_off_col_ptr,
        const aligned_vector<int>& recv_off_col_coarse,
        const aligned_vector<int>& on_col_ptr,
        const aligned_vector<int>& on_col_indices,
        const aligned_vector<int>& off_col_ptr,
        const aligned_vector<int>& off_col_indices,
        aligned_vector<int>& off_edgemark, 
        const aligned_vector<int>& off_proc_states, 
        aligned_vector<int>& off_proc_weight_updates)
{
    int start, end;
    int c, idx, idx_k;
    int on_nnz, off_nnz;
    int idx_start, idx_end;
    int new_idx;

    aligned_vector<int> on_sizes;
    aligned_vector<int> on_ptr(num_new_coarse+1);
    aligned_vector<int> on_indices;
    aligned_vector<int> off_sizes;
    aligned_vector<int> off_ptr(off_num_new_coarse+1);
    aligned_vector<int> off_indices;
    aligned_vector<int> on_proc_col_to_coarse;
    aligned_vector<int> off_proc_col_to_coarse;
    std::map<int, int> map_to_local;
    
    aligned_vector<int> c_dep_cache;
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

int find_off_proc_states(CommPkg* comm,
        const aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states,
        bool first_pass = false)
{
    int new_state;
    int num_new_coarse = 0;
    int off_proc_num_cols = off_proc_states.size();

    if (first_pass)
    {
        aligned_vector<int>& recvbuf = comm->communicate(states);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            new_state = recvbuf[i];
            if (new_state == 2)
            {
                num_new_coarse++;
            }
            off_proc_states[i] = new_state;
        }
    }
    else
    {
        aligned_vector<int>& recvbuf = ((ParComm*)comm)->conditional_comm(states, states, 
                off_proc_states, [&](const int a)
                {
                    return a == -1 || a == 2 || a == 3;
                });

        for (int i = 0; i < off_proc_num_cols; i++)
        {
            if (off_proc_states[i] == -1)
            {
                new_state = recvbuf[i];
                if (new_state == 2)
                {
                    num_new_coarse++;
                }
                off_proc_states[i] = new_state;
            }
        }
    }


    return num_new_coarse;
}

void find_off_proc_new_coarse(const ParCSRMatrix* S,
        CommPkg* comm,
        const std::map<int, int>& global_to_local,
        const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        const int* part_to_col,
        aligned_vector<int>& off_proc_col_ptr,
        aligned_vector<int>& off_proc_col_coarse,
        bool first_pass = false)
{
    int start, end;
    int idx, idx_k;
    int size;
    int idx_start, idx_end;
    int proc, count, buf_ptr;
    int ctr;
    int tag = 2999;
    int global_col;
    int n_sends = 0;
    int msg_avail;
    MPI_Status recv_status;

    aligned_vector<int> send_ptr;
    aligned_vector<int> send_buffer;
    aligned_vector<int> recv_buffer;

    off_proc_col_coarse.clear();
    int off_proc_num_cols = off_proc_states.size();

    if (first_pass)
    {
        send_ptr.resize(S->local_num_rows + 1);

        // For each row in send_data, find if any new coarse
        // points in row.  If so, add to buffer.  Let buffer_ptr
        // point to the beginning/end of each send_data msg
        send_ptr[0] = 0;
        for (int i = 0; i < S->local_num_rows; i++)
        {
            start = S->on_proc->idx1[i];
            end = S->on_proc->idx1[i+1];
            if (S->on_proc->idx2[idx_start] == idx)
            {
                start++;
            }
            for (int j = start; j < end; j++)
            {
                idx = S->on_proc->idx2[j];
                if (states[idx] == 2)
                {
                    send_buffer.push_back(S->on_proc_column_map[idx]);
                }
            }
            start = S->off_proc->idx1[i];
            end = S->off_proc->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                idx = S->off_proc->idx2[j];
                if (off_proc_states[idx] == 2)
                {
                    send_buffer.push_back(S->off_proc_column_map[idx]);
                }
            }
            send_ptr[i+1] =  send_buffer.size();
        }

        CSRMatrix* recv_mat = comm->communicate(send_ptr, send_buffer); 

        off_proc_col_ptr[0] = 0;
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            start = recv_mat->idx1[i];
            end = recv_mat->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                global_col = recv_mat->idx2[j];
                if (global_col >= S->partition->first_local_col 
                        && global_col <= S->partition->last_local_col)
                {
                    off_proc_col_coarse.push_back(part_to_col[global_col -
                            S->partition->first_local_col]);
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
            off_proc_col_ptr[i+1] = off_proc_col_coarse.size();
        }   
        delete recv_mat;
    }
    else
    {
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
                MPI_Isend(&(send_buffer[start]), end - start, MPI_INT, proc,
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

                if ((int) recv_buffer.size() < count)
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
                            off_proc_col_coarse.push_back(part_to_col[global_col -
                                    S->partition->first_local_col]);
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
}

void combine_weight_updates(CommPkg* comm,
        const aligned_vector<int>&states,
        const aligned_vector<int>& off_proc_states,
        const aligned_vector<int>& off_proc_weight_updates,
        aligned_vector<double>& weights,
        bool first_pass = false)
{
    std::function<double(double, int)> result_func = 
        [](const double a, const int b)
        {
            return a + b;
        };
    if (first_pass)
    {
        comm->communicate_T(off_proc_weight_updates, weights, result_func);
    }
    else
    {
        std::function<bool(int)> compare_func = [](const int a)
        {
            return a == -1;
        };        
        ((ParComm*)comm)->conditional_comm_T(off_proc_weight_updates, states, 
                off_proc_states, compare_func, weights, result_func);
    }
}

int update_states(aligned_vector<double>& weights, 
        aligned_vector<int>& states, const int remaining, aligned_vector<int>& unassigned)
{
    int num_states = states.size();

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
        else if (weights[u] < 1.0 || states[u] == 3)
        {
            weights[u] = 0.0;
            states[u] = 0;
        }
        else
        {
            unassigned[ctr++] = unassigned[i];
        }
    }

    return ctr;
}

void pmis_main_loop(ParCSRMatrix* S,
        aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states,
        bool tap_comm, double* rand_vals,
        data_t* comm_t)
{
    int start, end, row;
    int idx, ctr;
    int num_new_coarse;
    int num_remaining;
    int num_remaining_off;
    aligned_vector<double> off_proc_weights;
    aligned_vector<double> max_weights;
    aligned_vector<int> new_coarse_list;
    aligned_vector<int> unassigned;
    aligned_vector<int> unassigned_off;

    aligned_vector<int> on_col_ptr;
    aligned_vector<int> off_col_ptr;
    aligned_vector<int> on_col_indices;
    aligned_vector<int> off_col_indices;
    aligned_vector<double> weights;

    CommPkg* comm = S->comm;
    if (tap_comm) comm = S->tap_comm;

    if (S->local_num_rows)
    {
        weights.resize(S->local_num_rows);
        unassigned.resize(S->local_num_rows);
        max_weights.resize(S->local_num_rows);
        new_coarse_list.resize(S->local_num_rows);
    }
    if (S->off_proc_num_cols)
    {
        unassigned_off.resize(S->off_proc_num_cols);
        off_proc_weights.resize(S->off_proc_num_cols);
        off_proc_states.resize(S->off_proc_num_cols);
    }

    transpose(S, on_col_ptr, off_col_ptr, on_col_indices, off_col_indices);

    if (comm_t) *comm_t -= MPI_Wtime();
    initial_weights(S, comm, weights, rand_vals);
    if (comm_t) *comm_t += MPI_Wtime();

    // Find remaining vertices in on and off proc matrices
    num_remaining = 0;

    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == 1)
        {
            start = on_col_ptr[i];
            end = on_col_ptr[i+1];
            for (int j = start; j < end; j++)
            {
                row = on_col_indices[j];
                if (states[row] == -1)
                {
                    states[row] = 0;
                }
            }
        }
    }
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] == -1 && weights[i] < 1)
        {
            states[i] = 0;
	        weights[i] = 0.0;
        }
        else if (states[i] == -1)
        {
            unassigned[num_remaining++] = i;
        }
	    else 
        {
            weights[i] = 0.0;
        }
    }   
    
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<int>& recvbuf = comm->communicate(states);
    if (comm_t) *comm_t += MPI_Wtime();

    num_remaining_off = 0;
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        off_proc_states[i] = recvbuf[i];
        if (off_proc_states[i] == -1)
        {
            unassigned_off[num_remaining_off++] = i;
        }
    }

    // Find off_proc_weights
    bool first_pass = true;

    if (comm_t) *comm_t -= MPI_Wtime();
    find_off_proc_weights(comm, states, off_proc_states, 
            weights, off_proc_weights, first_pass);
    if (comm_t) *comm_t += MPI_Wtime();

    while (num_remaining || num_remaining_off)
    {
        // Find max unassigned weight in each row / column
        if (comm_t) *comm_t -= MPI_Wtime();
        find_max_off_weights(comm, off_col_ptr, off_col_indices, 
                states, off_proc_states, weights, max_weights, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();
        
        // For each vertex, if max in neighborhood, add to C
        num_new_coarse = select_independent_set(S, num_remaining, unassigned,
                weights, off_proc_weights, max_weights, on_col_ptr,
                on_col_indices, states, off_proc_states, new_coarse_list);

        if (comm_t) *comm_t -= MPI_Wtime();
        find_off_proc_states(comm, states, off_proc_states, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();

        // For each row, if new C point in row, add row to F
        for (int i = 0; i < num_new_coarse; i++)
        {
            idx = new_coarse_list[i];
            start = on_col_ptr[idx];
            end = on_col_ptr[idx+1];
            for (int j = start; j < end; j++)
            {
                row = on_col_indices[j];
                if (states[row] == -1)
                {
                    states[row] = 3;
                }
            }
        }
        for (int i = 0; i < num_remaining_off; i++)
        {
            idx = unassigned_off[i];
            if (off_proc_states[idx] == 2)
            {
                start = off_col_ptr[idx];
                end = off_col_ptr[idx+1];
                for (int j = start; j < end; j++)
                {
                    row = off_col_indices[j];
                    if (states[row] == -1)
                    {
                        states[row] = 3;
                    }
                }
            }       
        }

        if (comm_t) *comm_t -= MPI_Wtime();
        find_off_proc_states(comm, states, off_proc_states, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();

        num_remaining = update_states(weights, states, num_remaining, unassigned);
        num_remaining_off = update_states(off_proc_weights,
               off_proc_states, num_remaining_off, unassigned_off);

        first_pass = false;
        comm = S->comm;
    }
}

void cljp_main_loop(ParCSRMatrix* S,
        aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states,
        bool tap_comm, double* rand_vals,
        data_t* comm_t, data_t* comm_mat_t)
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
    int remaining, off_remaining;
    int size, global_col;

    CommPkg* comm = S->comm;
    if (tap_comm) comm = S->tap_comm;

    aligned_vector<double> max_weights;
    aligned_vector<int> weight_updates;
    aligned_vector<double> off_proc_weights;
    aligned_vector<int> off_proc_col_coarse;
    aligned_vector<int> off_proc_weight_updates;
    aligned_vector<int> off_proc_col_ptr;
    std::map<int, int> global_to_local;
    aligned_vector<int> new_coarse_list;
    aligned_vector<int> off_new_coarse_list;
    aligned_vector<int> unassigned;
    aligned_vector<int> unassigned_off;
    aligned_vector<int> on_col_ptr;
    aligned_vector<int> off_col_ptr;
    aligned_vector<int> on_col_indices;
    aligned_vector<int> off_col_indices;
    aligned_vector<int> on_edgemark;
    aligned_vector<int> off_edgemark;
    aligned_vector<double> weights;

    int count;
    int n_sends, n_recvs;
    int prev_ctr;
    int msg_avail;
    MPI_Status recv_status;

    int* part_to_col = S->map_partition_to_local();

    if (S->local_num_rows)
    {
        weights.resize(S->local_num_rows, 0);
        max_weights.resize(S->local_num_rows);
        new_coarse_list.resize(S->local_num_rows);
        unassigned.resize(S->local_num_rows);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_weight_updates.resize(S->off_proc_num_cols);
        off_proc_weights.resize(S->off_proc_num_cols, 0);
        off_proc_states.resize(S->off_proc_num_cols);
        off_new_coarse_list.resize(S->off_proc_num_cols);
        unassigned_off.resize(S->off_proc_num_cols);
    }
    if (S->on_proc->nnz)
    {
        on_edgemark.resize(S->on_proc->nnz, 1);
    }
    if (S->off_proc->nnz)
    {
        off_edgemark.resize(S->off_proc->nnz, 1);
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<int>& recvbuf = comm->communicate(states);
    if (comm_t) *comm_t += MPI_Wtime();

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        off_proc_states[i] = recvbuf[i];
    }
    off_proc_col_ptr.resize(S->off_proc_num_cols + 1);

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        global_to_local[S->off_proc_column_map[i]] = i;
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    initial_weights(S, comm, weights, rand_vals);
    if (comm_t) *comm_t += MPI_Wtime();

    transpose(S, on_col_ptr, off_col_ptr, on_col_indices, off_col_indices);

    remaining = 0;
    off_remaining = 0;
    num_new_coarse = 0;
    for (int i = 0; i < S->on_proc_num_cols; i++)
    {
        if (states[i] == -1)
        {
            unassigned[remaining++] = i;
        }
	    else
	    {
            weights[i] = 0.0;
            
            if (states[i] == 1)
            {
                new_coarse_list[num_new_coarse++] = i;
            }
	    }
    }

    if (num_new_coarse)
    {
        update_row_weights(S, num_new_coarse, new_coarse_list,
                on_edgemark, off_edgemark, states, off_proc_states,
                weights, off_proc_weight_updates);
        update_local_dist2_weights(S, num_new_coarse, new_coarse_list,
                0, off_new_coarse_list, on_col_ptr, on_col_indices,
                off_col_ptr, off_col_indices, on_edgemark, states, weights);
    }

    bool first_pass = true;
    /**********************************************
     * Find weights of unassigned neighbors (off proc cols)
     **********************************************/
    if (comm_t) *comm_t -= MPI_Wtime();
    find_off_proc_weights(comm, states, off_proc_states, 
            weights, off_proc_weights, first_pass);
    if (comm_t) *comm_t += MPI_Wtime();


    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] == -1)
        {
            unassigned_off[off_remaining++] = i;
        }
	    else
	    {
            off_proc_weights[i] = 0.0;
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
        if (comm_t) *comm_t -= MPI_Wtime();
        find_max_off_weights(comm, off_col_ptr, off_col_indices, 
                states, off_proc_states, weights, max_weights, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();

        /**********************************************
        * Select independent set: all indices with
        * maximum weight among unassigned neighbors
        **********************************************/
        num_new_coarse = select_independent_set(S, remaining, unassigned,
                weights, off_proc_weights, max_weights, on_col_ptr, 
                on_col_indices, states, off_proc_states, new_coarse_list);

        // Communicate updated states to neighbors
        // Only communicating previously unassigned states
        if (comm_t) *comm_t -= MPI_Wtime();
        off_num_new_coarse = find_off_proc_states(comm, states, off_proc_states, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();

        ctr = 0;
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            if (off_proc_states[i] == 2)
            {
                off_new_coarse_list[ctr++] = i;
            }
        }

        // Find new coarse influenced by each off_proc col
        // TODO -- Add first pass option
        if (comm_mat_t) *comm_mat_t -= MPI_Wtime();
        find_off_proc_new_coarse(S, comm, global_to_local, states, off_proc_states, 
                part_to_col, off_proc_col_ptr, off_proc_col_coarse, first_pass);
        if (comm_mat_t) *comm_mat_t += MPI_Wtime();

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
        if (comm_t) *comm_t -= MPI_Wtime();
        combine_weight_updates(comm, states, off_proc_states,
                off_proc_weight_updates, weights, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();

        // Find weights of unassigned neighbors (off proc cols)
        if (comm_t) *comm_t -= MPI_Wtime();
        find_off_proc_weights(comm, states, off_proc_states, 
                weights, off_proc_weights, first_pass);
        if (comm_t) *comm_t += MPI_Wtime();

        // Update states, changing any new coarse states
        // from 2 to 1 (and changes weight to 0) and
        // set state of any unassigned indices with weight
        // less than 1 to fine (also update off_proc_states)
        remaining = update_states(weights, states, remaining, unassigned);
        off_remaining = update_states(off_proc_weights,
                off_proc_states, off_remaining, unassigned_off);

        first_pass = false;
        comm = S->comm;
    }

    delete[] part_to_col;
}

