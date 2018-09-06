// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "cf_splitting.hpp"

// TODO - parts of cf_splitting were taken from pyamg... how to cite this?

using namespace raptor;

void transpose(const CSRMatrix* S,
        aligned_vector<int>& col_ptr,
        aligned_vector<int>& col_indices)
{
    int start, end;
    int col, idx;
    aligned_vector<int> col_sizes;

    col_ptr.resize(S->n_cols + 1);
    if (S->n_cols)
    {
        col_sizes.resize(S->n_cols, 0);
    }
    if (S->nnz)
    {
        col_indices.resize(S->nnz);
    }

    // Calculate nnz in each col
    for (int i = 0; i < S->n_rows; i++)
    {
        start = S->idx1[i];
        end = S->idx1[i+1];
        if (S->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            col = S->idx2[j];
            col_sizes[col]++;
        }
    }

    // Create col_ptr
    col_ptr[0] = 0;
    for (int i = 0; i < S->n_cols; i++)
    {
        col_ptr[i+1] = col_ptr[i] + col_sizes[i];
        col_sizes[i] = 0;
    }

    // Add indices to col_indices
    for (int i = 0; i < S->n_rows; i++)
    {
        start = S->idx1[i];
        end = S->idx1[i+1];
        if (S->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            col = S->idx2[j];
            idx = col_ptr[col] + col_sizes[col]++;
            col_indices[idx] = i;
        }
    }
}

void rs_first_pass(const CSRMatrix* S,
        const aligned_vector<int>& col_ptr,
        const aligned_vector<int>& col_indices,
        aligned_vector<int>& weights,
        aligned_vector<int>& states)
{
    int start, end;
    int idx_start, idx_end;
    int idx, idx_k, col;
    int old_pos, new_pos;
    int weight, weight_k;

    aligned_vector<int> weight_ptr(S->n_cols + 1, 0);
    aligned_vector<int> weight_sizes(S->n_cols);
    aligned_vector<int> weight_idx_to_col(S->n_cols);
    aligned_vector<int> col_to_weight_idx(S->n_cols);

    // Find number of each weight
    for (int i = 0; i < S->n_cols; i++)
    {
        weight_sizes[weights[i]]++;
    }

    // Find first index of each weight
    weight_ptr[0] = 0;
    for (int i = 0; i < S->n_cols; i++)
    {
        weight_ptr[i+1] = weight_ptr[i] + weight_sizes[i];
        weight_sizes[i] = 0;
    }

    // Find a col at each index, and index of each col
    for (int i = 0; i < S->n_cols; i++)
    {
        weight = weights[i];
        idx = weight_ptr[weight] + weight_sizes[weight]++;
        weight_idx_to_col[idx] = i;
        col_to_weight_idx[i] = idx;
    }

    // Find C/F points (starting with largest weights)
    for (int i = S->n_cols - 1; i >= 0; i--)
    {
        col = weight_idx_to_col[i];
        weight = weights[col];
        weight_sizes[weight]--;

        if (states[col] != -1)
        {
            continue;
        }
        else
        {
            states[col] = 1;

            start = col_ptr[col];
            end = col_ptr[col+1];
            for (int j = start; j < end; j++)
            {
                idx = col_indices[j];
                if (states[idx] == -1)
                {
                    states[idx] = 0;

                    idx_start = S->idx1[idx];
                    idx_end = S->idx1[idx+1];
                    if (S->idx2[idx_start] == idx)
                    {
                        idx_start++;
                    }
                    for (int k = idx_start; k < idx_end; k++)
                    {
                        idx_k = S->idx2[k];
                        if (states[idx_k] == -1)
                        {
                            // Increment weight
                            weight_k = weights[idx_k];
                            if (weight_k >= S->n_cols - 1)
                            {
                                continue;
                            }

                            // Move col to end of weight interval (incremented
                            // weight, so now belongs in next interval)
                            old_pos = col_to_weight_idx[idx_k];
                            new_pos = weight_ptr[weight_k] + weight_sizes[weight_k] - 1;
                            col_to_weight_idx[weight_idx_to_col[old_pos]] = new_pos;
                            col_to_weight_idx[weight_idx_to_col[new_pos]] = old_pos;
                            std::swap(weight_idx_to_col[old_pos], weight_idx_to_col[new_pos]);

                            weight_sizes[weight_k] -= 1;
                            weight_sizes[weight_k+1] += 1;
                            weight_ptr[weight_k+1] = new_pos;

                            // Increment weight of dist-2 connection
                            weights[idx_k]++;
                        }
                    }
                }
            }

            start = S->idx1[col];
            end = S->idx1[col+1];
            if (S->idx2[start] == col)
            {
                start++;
            }
            for (int j = start; j < end; j++)
            {
                idx = S->idx2[j];

                if (states[idx] == -1)
                {
                    // If no weight to decrement, continue
                    weight = weights[idx];
                    if (weight == 0)
                    {
                        continue;
                    }

                    // Move idx to beginning of interval (decremented weight)
                    old_pos = col_to_weight_idx[idx];
                    new_pos = weight_ptr[weight];

                    col_to_weight_idx[weight_idx_to_col[old_pos]] = new_pos;
                    col_to_weight_idx[weight_idx_to_col[new_pos]] = old_pos;
                    std::swap(weight_idx_to_col[old_pos], weight_idx_to_col[new_pos]);

                    // Update intervals
                    weight_sizes[weight] -= 1;
                    weight_sizes[weight-1] += 1;
                    weight_ptr[weight] += 1;
                    weight_ptr[weight-1] = weight_ptr[weight] - weight_sizes[weight-1];

                    // Decrement weight
                    weights[idx]--;
                }
            }
        }
    }
}

void rs_second_pass(const CSRMatrix* S,
        aligned_vector<int>& weights,
        aligned_vector<int>& states)
{
    int start, end, col;
    int start_k, end_k, col_k;
    bool connection;

    aligned_vector<int> row_coarse(S->n_rows, -1);

    for (int i = 0; i < S->n_rows; i++)
    {
        if (states[i] == 1) continue;

        start = S->idx1[i];
        end = S->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = S->idx2[j];
            if (states[col] == 1) 
                row_coarse[col] = i;
        }

        for (int j = start; j < end; j++)
        {
            col = S->idx2[j];
            if (states[col] == 0)
            {
                start_k = S->idx1[col];
                end_k = S->idx1[col+1];
                connection = false;
                if (start_k == end_k) continue;
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = S->idx2[k];
                    if (row_coarse[col_k] == i)
                    {
                        connection = true;
                        break;
                    }
                }
                
                if (!connection)
                {
                    row_coarse[col] = i;
                    states[col] = 1;
                }
            }
        }
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
 ***** S : CSRMatrix*
 *****    Strength of connection matrix
 **************************************************************/
void split_rs(CSRMatrix* S,
        aligned_vector<int>& states, bool prev_states, bool second_pass)
{
    int start, end;
    

    if (!S->diag_first)
    {
        S->move_diag();
    }

    aligned_vector<int> col_ptr;
    aligned_vector<int> col_indices;
    aligned_vector<int> weights;
    if (S->n_rows)
    {
        weights.resize(S->n_rows);
    }

    if (!prev_states && S->n_rows)
    {
        states.resize(S->n_rows);
        for (int i = 0; i < S->n_rows; i++)
        {
            states[i] = -1;
        }
    }

    // Find CSC sparsity pattern of S
    transpose(S, col_ptr, col_indices);

    // Initialize weights for RugeStuben cf splitting
    for (int i = 0; i < S->n_cols; i++)
    {
        start = col_ptr[i];
        end = col_ptr[i+1];
        weights[i] = end - start;
    }

    rs_first_pass(S, col_ptr, col_indices, weights, states);

    if (second_pass)
    {
        rs_second_pass(S, weights, states);
    }
}

int select_independent_set(CSRMatrix* S, aligned_vector<int>& col_ptr,
        aligned_vector<int>& col_indices, int remaining, aligned_vector<int>& unassigned,
        aligned_vector<int>& states, aligned_vector<double>& weights,
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

        start = S->idx1[u];
        end = S->idx1[u+1];
        if (S->idx2[start] == u)
        {
            start++;
        }
        for (j = start; j < end; j++)
        {
            idx = S->idx2[j];
            if (weights[idx] > weight)
            {
                break;
            }
        }
        if (j != end)
        {
            continue;
        }

        start = col_ptr[u];
        end = col_ptr[u+1];
        for (j = start; j < end; j++)
        {
            idx = col_indices[j];
            if (weights[idx] > weight)
            {
                break;
            }
        }
        if (j != end)
        {
            continue;
        }
        
        states[u] = 2;
        new_coarse_list[num_new_coarse++] = u;
    }

    return num_new_coarse;
}

void update_weights(CSRMatrix* S, aligned_vector<int>& col_ptr, aligned_vector<int>& col_indices, 
        aligned_vector<int>& edgemark, aligned_vector<int>& c_dep_cache, 
        aligned_vector<int>& new_coarse_list, int num_new_coarse, 
        aligned_vector<int>& states, aligned_vector<double>& weights)
{
    int start, end;
    int idx, idx_k, c;
    int idx_start, idx_end;
    

    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];
        start = S->idx1[c];
        end = S->idx1[c+1];
        if (S->idx2[start] == c)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->idx2[j];
            if (states[idx] == -1 && edgemark[j])
            {
                edgemark[j] = 0;
                weights[idx]--;
            }
        }
    }

    for (int i = 0; i < num_new_coarse; i++)
    {
        c = new_coarse_list[i];
        start = col_ptr[c];
        end = col_ptr[c+1];
        for (int j = start; j < end; j++)
        {
            idx = col_indices[j];
            if (states[idx] == -1)
            {
                c_dep_cache[idx] = c;
            }
        }

        for (int j = start; j < end; j++)
        {
            idx = col_indices[j];
            if (states[idx] == 1) continue;

            idx_start = S->idx1[idx];
            idx_end = S->idx1[idx+1];
            if (S->idx2[idx_start] == idx)
            {
                idx_start++;
            }
            for (int k = idx_start; k < idx_end; k++)
            {
                idx_k = S->idx2[k];
                if (states[idx_k] == -1 && edgemark[k] && c_dep_cache[idx_k] == c)
                {
                    edgemark[k] = 0;
                    weights[idx_k]--;
                }
            }
        }

    }
}

int update_states(int remaining, aligned_vector<int>& unassigned, aligned_vector<int>& states,
        aligned_vector<double>& weights)
{
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
        }
        else
        {
            unassigned[ctr++] = u;
        }
    }

    return ctr;
}

void cljp_main_loop(CSRMatrix* S, aligned_vector<int>& col_ptr, aligned_vector<int>& col_indices,
        aligned_vector<int>& states, double* rand_vals = NULL)
{
    int num_new_coarse;
    int remaining;
    int start, end;
    
    int idx;
    aligned_vector<int> edgemark;
    aligned_vector<double> weights;
    aligned_vector<int> new_coarse_list;
    aligned_vector<int> unassigned;
    aligned_vector<int> c_dep_cache;
    if (S->n_rows)
    {
        weights.resize(S->n_rows);
        new_coarse_list.resize(S->n_rows);
        c_dep_cache.resize(S->n_rows);
        unassigned.resize(S->n_rows);
        std::iota(unassigned.begin(), unassigned.end(), 0);
    }
    if (S->nnz)
    {
        edgemark.resize(S->nnz, 1);
    }

 
    //TODO -- change to random... reading for testing
    if (rand_vals)
    {
        for (int i = 0; i < S->n_rows; i++)
        {
            weights[i] = rand_vals[i];
        }
    }
    else
    {
        srand(time(NULL));
        for (int i = 0; i < S->n_rows; i++)
        {
            // Random value [0,1)
            weights[i] = ((double)(rand())) / RAND_MAX;
        }
    }

    for (int i = 0; i < S->n_rows; i++)
    {
        start = S->idx1[i];
        end = S->idx1[i+1];
        if (S->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            idx = S->idx2[j];
            weights[idx] += 1;
        }
    }

    remaining = S->n_rows;
    while (remaining)
    {
        // Select independent set
        num_new_coarse = select_independent_set(S, col_ptr, col_indices, 
                remaining, unassigned,
                states, weights, new_coarse_list);

        // Update weights
        update_weights(S, col_ptr, col_indices, edgemark, c_dep_cache,
                new_coarse_list, num_new_coarse, states, weights);

        // Update states
        remaining = update_states(remaining, unassigned, states, weights);
    }
}

void pmis_main_loop(CSRMatrix* S, aligned_vector<int>& col_ptr, aligned_vector<int>& col_indices,
        aligned_vector<int>& states, double* rand_vals)
{
    int num_new_coarse;
    int start, end, col, row;
    int ctr, idx;
    int num_remaining;
    double row_weight, col_weight;
    double max_row_weight, max_col_weight;
    aligned_vector<double> weights;
    aligned_vector<int> unassigned;
    aligned_vector<int> new_coarse_list;

    if (S->n_rows)
    {
        weights.resize(S->n_rows);
        unassigned.resize(S->n_rows);
        new_coarse_list.resize(S->n_rows);
    }

    // Assign random weight to each vertex
    if (rand_vals)
    {
        for (int i = 0; i < S->n_rows; i++)
        {
            weights[i] = rand_vals[i];
        }
    }
    else
    {
        srand(102483);
        for (int i = 0; i < S->n_rows; i++)
        {
            // Random value [0,1)
            weights[i] = ((double)(rand())) / RAND_MAX;
        }
    }

    // Update vertex weights (number of rows in which it is a column)
    for (int i = 0; i < S->n_rows; i++)
    {
        start = S->idx1[i];
        end = S->idx1[i+1];
        if (S->idx2[start] == i)
        {
            start++;
        }
        for (int j = start; j < end; j++)
        {
            col = S->idx2[j];
            weights[col] += 1;
        }
    }

    num_remaining = 0;
    for (int i = 0; i < S->n_rows; i++)
    {
        if (weights[i] < 1)
        {
            states[i] = 0;
        }
        else
        {
            unassigned[num_remaining++] = i;
        }
    }

    while (num_remaining > 0)
    {
        // Find max unassigned weight in each row / column
        num_new_coarse = select_independent_set(S, col_ptr, col_indices,
                num_remaining, unassigned, states, weights, new_coarse_list);

        for (int i = 0; i < num_new_coarse; i++)
        {
            idx = new_coarse_list[i];
            start = col_ptr[idx];
            end = col_ptr[idx+1];
            for (int j = start; j < end; j++)
            {
                row = col_indices[j];
                if (states[row] == -1)
                {
                    states[row] = 0;
                    weights[row] = 0;
                }
            }
        }

        num_remaining = update_states(num_remaining, unassigned, states, weights);
    }
}

void split_cljp(CSRMatrix* S, 
        aligned_vector<int>& states,
        double* rand_vals)
{
    aligned_vector<int> col_ptr;
    aligned_vector<int> col_indices;

    if (!S->diag_first)
    {
        S->move_diag();
    }

    if (S->n_rows)
    {
        states.resize(S->n_rows);
    }

    // Find column-wise sparsity pattern of S
    transpose(S, col_ptr, col_indices);

    // Set initial states to undecided (-1)
    for (int i = 0; i < S->n_rows; i++)
    {
        states[i] = -1;
    }

    cljp_main_loop(S, col_ptr, col_indices, states, rand_vals);
}


void split_pmis(CSRMatrix* S, aligned_vector<int>& states, double* rand_vals)
{
    aligned_vector<int> col_ptr;
    aligned_vector<int> col_indices;

    if (!S->diag_first)
    {
        S->move_diag();
    }
    if (S->n_rows)
    {
        states.resize(S->n_rows);
    }

    // Find column-wise sparsity pattern of S
    transpose(S, col_ptr, col_indices);

    for (int i = 0; i < S->n_rows; i++)
    {
        states[i] = -1;
    }

    pmis_main_loop(S, col_ptr, col_indices, states, rand_vals);
}


