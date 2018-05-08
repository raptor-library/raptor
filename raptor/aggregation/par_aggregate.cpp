// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aggregation/par_aggregate.hpp"

int aggregate(ParCSRMatrix* A, ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, aligned_vector<int>& aggregates,
        aligned_vector<int>& off_proc_aggregates, double* rand_vals)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (S->local_num_rows == 0)
        return 0;

    // Initialize Variables
    aligned_vector<int> off_proc_S_to_A;
    int n_aggs = 0;
    int off_aggs = 0;
    int start, end, col;
    int col_A, global_col;
    int ctr, max_agg, j;
    double max_val, val;

    aligned_vector<double> r;
    aligned_vector<double> off_proc_r;


    if (S->local_num_rows)
    {
        aggregates.resize(S->local_num_rows, -1);
        r.resize(S->local_num_rows);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_aggregates.resize(S->off_proc_num_cols, -1);
        off_proc_r.resize(S->off_proc_num_cols);
    }

    if (rand_vals)
    {
        for (int i = 0; i < S->local_num_rows; i++)
        {
            r[i] = rand_vals[i];
        }
    }
    else
    {
        for (int i = 0; i < S->local_num_rows; i++)
        {
            r[i] = ((double)(rand()) / RAND_MAX);
        }
    }
    aligned_vector<double>& rands = S->comm->communicate(r);
    std::copy(rands.begin(), rands.end(), off_proc_r.begin());

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
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] > 0)
        {
            aggregates[i] = S->on_proc_column_map[i];
            n_aggs++;
        }
    }
    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        if (off_proc_states[i] > 0)
        { 
            off_proc_aggregates[i] = S->off_proc_column_map[i];
        }
    }

    // Pass 1 : add each node to neighboring aggregate
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] > 0) continue;

        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        for (j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];

            if (states[col] > 0)
            {
                aggregates[i] = aggregates[col]; // global col
                break;
            }
        }
        if (j == end)
        {
            start = S->off_proc->idx1[i];
            end = S->off_proc->idx1[i+1];
            for (j = start; j < end; j++)
            {
                col = S->off_proc->idx2[j];
                if (off_proc_states[col] > 0)
                {
                    aggregates[i] = off_proc_aggregates[col]; // global col
                    break;
                }
            }
        }
    }

    // Communicate aggregates (global rows)
    aligned_vector<int>& recvbuf = S->comm->communicate(aggregates);
    std::copy(recvbuf.begin(), recvbuf.end(), off_proc_aggregates.begin());

    // Pass 2 : add remaining aggregate to that of strongest neighbor
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (aggregates[i] >= 0) continue;

        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        ctr = A->on_proc->idx1[i];
        max_val = 0.0;
        max_agg = - A->partition->global_num_rows; 
        for (j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];
            while (A->on_proc->idx2[ctr] != col)
                ctr++;
            val = fabs(A->on_proc->vals[ctr]) + r[col];
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
            ctr = A->off_proc->idx1[i];
            global_col = S->off_proc_column_map[col];
            while (A->off_proc_column_map[A->off_proc->idx2[ctr]] != global_col)
                ctr++;
            val = fabs(A->off_proc->vals[ctr]) + off_proc_r[col];
            if (val > max_val && off_proc_aggregates[col] >= 0)
            {
                max_val = val;
                max_agg = off_proc_aggregates[col];
            }
        }

        aggregates[i] = - (max_agg + 1);
    }

    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (aggregates[i] < 0)
            aggregates[i] = - (aggregates[i] + 1);
    }

    // Communicate aggregates (global rows)
    recvbuf = S->comm->communicate(aggregates);
    std::copy(recvbuf.begin(), recvbuf.end(), off_proc_aggregates.begin());

    return n_aggs;
}

