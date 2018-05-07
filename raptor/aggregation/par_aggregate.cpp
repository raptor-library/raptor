// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aggregation/par_aggregate.hpp"

int aggregate(ParCSRMatrix* A, ParCSRMatrix* S, std::vector<int>& states,
        std::vector<int>& off_proc_states, std::vector<int>& aggregates,
        std::vector<int>& off_proc_aggregates)
{
    if (A->local_num_rows == 0)
        return 0;

    // Initialize Variables
    std::vector<int> aggregates(A->local_num_rows, -1);
    std::vector<int> off_proc_S_to_A;
    int n_aggs = 0;
    int off_aggs = 0;
    int start, end, col, global_col;
    int ctr, max_agg, j;
    double max_val;

    if (S->off_proc_num_cols)
    {
        off_proc_S_to_A.resize(S->off_proc_num_cols);
        ctr = 0;
        for (int i = 0; i < S->off_proc_num_cols; i++)
        {
            global_col = S->off_proc_column_map[i];
            while (A->off_proc_column_map[ctr] != global_col)
                ctr++;
            off_proc_S_to_A[i] = ctr;
        }
    }

    // Label aggregates as global column
    for (int i = 0; i < A->local_num_rows; i++)
    {
        if (states[i] > 0)
        {
            aggregates[i] = A->on_proc_column_map[i];
            n_aggs++;
        }
    }
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] > 0)
            off_proc_aggregates[i] = A->off_proc_column_map[i];
    }

    // Pass 1 : add each node to neighboring aggregate
    for (int i = 0; i < A->n_rows; i++)
    {
        if (states[i]) continue;

        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (j = start; j < end; j++)
        {
            col = A->on_proc->idx2[j];
            if (states[col] > 0)
            {
                aggregates[i] = aggregates[col]; // global col
                break;
            }
        }
        if (j == end)
        {
            start = A->off_proc->idx1[i];
            end = A->off_proc->idx1[i+1];
            for (j = start; j < end; j++)
            {
                col = A->off_proc->idx2[j];
                if (off_proc_states[col] > 0)
                {
                    aggregates[i] = off_proc_aggregates[col]; // global col
                    break;
                }
            }
        }
    }

    // Communicate aggregates (global rows)
    std::vector<int>& recvbuf = A->comm->communicate(aggregates);
    std::copy(recvbuf.begin(), recvbuf.end(), off_proc_aggregates.begin();

    // Pass 2 : add remaining aggregate to that of strongest neighbor
    for (int i = 0; i < A->n_rows; i++)
    {
        if (aggregates[i] >= 0) continue;

        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        ctr = A->on_proc->idx1[i];
        max_val = 0.0;
        max_agg = -3; 
        for (j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            while (A->on_proc->idx2[ctr] != col)
                ctr++;
            val = A->on_proc->vals[ctr];

            if (val > max_val && aggregates[col] >= 0)
            {
                max_val = val;
                max_agg = aggregates[col];
            }
        }

        start = S->off_proc->idx1[i];
        end = S->off_proc->idx1[i+1];
        ctr = A->off_proc->idx1[i];
        for (j = start; j < end; j++)
        {
            col = S->off_proc->idx2[j];
            global_col = S->off_proc_column_map[col];
            while (A->off_proc_column_map[A->off_proc->idx2[ctr]] != global_col)
                ctr++;
            val = A->off_proc->vals[j];
            
            if (val > max_val && off_proc_aggregates[col] >= 0)
            {
                max_val = val;
                max_agg = off_proc_aggregates[col];
            }
        }

        aggregates[i] = max_agg;
    }

    // Communicate aggregates (global rows)
    recvbuf = A->comm->communicate(aggregates);
    std::copy(recvbuf.begin(), recvbuf.end(), off_proc_aggregates.begin();

    return n_aggs;
}

