// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aggregation/par_aggregate.hpp"

int aggregate(ParCSRMatrix* A, ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, aligned_vector<int>& aggregates,
        bool tap_comm, double* rand_vals, data_t* comm_t)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    S->sort();
    S->on_proc->move_diag();
    A->sort();
    A->on_proc->move_diag();

    // Initialize Variables
    aligned_vector<int> off_proc_S_to_A;
    int n_aggs = 0;
    int off_aggs = 0;
    int start, end, col;
    int col_A, global_col;
    int ctr, max_agg, j;
    double max_val, val;

    aligned_vector<int> off_proc_aggregates;
    aligned_vector<double> r;
    aligned_vector<double> off_proc_r;

    CommPkg* comm = S->comm;
    if (tap_comm)
    {
        comm = S->tap_comm;
    }

    if (S->local_num_rows)
    {
        aggregates.resize(S->local_num_rows, -1);
        r.resize(S->local_num_rows, 0);
    }
    if (S->off_proc_num_cols)
    {
        off_proc_aggregates.resize(S->off_proc_num_cols, -1);
        off_proc_r.resize(S->off_proc_num_cols, 0);
    }

    if (rand_vals)
    {
        for (int i = 0; i < S->local_num_rows; i++)
        {
            r[i] = rand_vals[i];
        }
    }
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& rands = comm->communicate(r);
    std::copy(rands.begin(), rands.end(), off_proc_r.begin());
    if (comm_t) *comm_t += MPI_Wtime();

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
        if (S->on_proc->idx1[i+1] - S->on_proc->idx1[i] <= 1 
                   && S->off_proc->idx1[i+1] == S->off_proc->idx1[i])
        {
            aggregates[i] = - A->partition->global_num_rows;
        }
        else if (states[i] == Selected)
        {
            aggregates[i] = S->on_proc_column_map[i];
            n_aggs++;
        }
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<int>& init_aggs = comm->communicate(aggregates);
    if (comm_t) *comm_t += MPI_Wtime();

    for (int i = 0; i < S->off_proc_num_cols; i++)
    {
        off_proc_aggregates[i] = init_aggs[i];
    }

    // Pass 1 : add each node to neighboring aggregate
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (states[i] == Selected) continue;

        start = S->on_proc->idx1[i];
        end = S->on_proc->idx1[i+1];
        for (j = start; j < end; j++)
        {
            col = S->on_proc->idx2[j];

            if (states[col] == Selected && aggregates[col] >= 0)
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
                if (off_proc_states[col] == Selected && off_proc_aggregates[col] >= 0)
                {
                    aggregates[i] = off_proc_aggregates[col]; // global col
                    break;
                }
            }
        }
    }

    // Communicate aggregates (global rows)
    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<int>& recvbuf = comm->communicate(aggregates);
    if (comm_t) *comm_t += MPI_Wtime();

    std::copy(recvbuf.begin(), recvbuf.end(), off_proc_aggregates.begin());

    // Pass 2 : add remaining aggregate to that of strongest neighbor
    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (aggregates[i] == -1) 
        {
            start = S->on_proc->idx1[i];
            end = S->on_proc->idx1[i+1];
            ctr = A->on_proc->idx1[i];
            max_val = 0.0;
            max_agg = -A->partition->global_num_rows; 
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
    }

    for (int i = 0; i < S->local_num_rows; i++)
    {
        if (aggregates[i] <= -A->partition->global_num_rows)
            aggregates[i] = -1;
        else if (aggregates[i] < 0)
            aggregates[i] = - (aggregates[i] + 1);
    }

    return n_aggs;
}

