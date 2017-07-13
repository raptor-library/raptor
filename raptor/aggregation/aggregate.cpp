// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;
// Active == -1
// F == 0
// C == 1
//
// TODO -- currently communicating all of local_states at each iteration... this
// is not necessary (only need to communicate updates)

int ParCSRMatrix::maximal_independent_set(std::vector<int>& local_states,
        std::vector<int>& off_proc_states, int max_iters)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start, end, col, row;
    int n_sends, proc;
    int count;
    int num_agg = 0;
    int num_iters = 0;
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int row_size, j;
    int num_coarse = 0;
    int tag = 3542;
    bool active = true;
    bool make_coarse;

    int col_state, row_state;
    double col_rand, row_rand;

    int finished, msg_avail;
    MPI_Request barrier_request;
    MPI_Status recv_status;

    std::vector<int> buffer;
    if (comm->send_data->size_msgs)
        buffer.resize(comm->send_data->size_msgs);

    // Initialize states of all rows / columns to -1
    if (local_num_rows)
    {
        local_states.resize(local_num_rows, -1);
    }
    if (off_proc_num_cols)
    {
        off_proc_states.resize(off_proc_num_cols, -1);
    }

    // Find random value associated with each local row, and send 
    // to all processes that hold non-zeros in associated columns
    std::vector<double> local_rands;
    std::vector<double> off_proc_rands;
    if (local_num_rows)
    {
        local_rands.resize(local_num_rows);
        for (int i = 0; i < local_num_rows; i++)
        {
            local_rands[i] = ((double) rand() / RAND_MAX);
        }
    }
    comm->communicate(local_rands.data());
    if (off_proc_num_cols)
    {
        off_proc_rands.resize(off_proc_num_cols);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            off_proc_rands[i] = comm->recv_data->buffer[i];
        }
    }

    // Create proc_recv_starts, which maps proc recv to 
    // position in recv_buffer to which recv is stored
    std::vector<int> proc_recv_starts(num_procs);
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        proc_recv_starts[proc] = start;
    }

    // Iterate through rows, changing local_states to C/F
    while (active && (max_iters == -1 || num_iters < max_iters))
    {
        active = false;
        num_iters++;
        for (int row = 0; row < local_num_rows; row++)
        {
            make_coarse = false;
            row_rand = local_rands[row];
            row_state = local_states[row];

            if (row_state != -1)
            {
                continue;
            }

            // Assuming diagonal value is not zero...
            row_start_on = on_proc->idx1[row]; // +1 to skip diag
            row_end_on = on_proc->idx1[row+1];
            for (j = row_start_on; j < row_end_on; j++)
            {
                col = on_proc->idx2[j];
                col_state = local_states[col];
                if (col_state == 1)
                {
                    local_states[row] = 0;
                    break;
                }
                if (col_state == -1)
                {
                    col_rand = local_rands[col];
                    if (col_rand > row_rand) 
                    {
                        break;
                    }
                    else if (col_rand == row_rand && col > row)
                    {
                        break;
                    }
                }
            }
            if (j == row_end_on)
            {
                row_start_off = off_proc->idx1[row];
                row_end_off = off_proc->idx1[row+1];
                for (j = row_start_off; j < row_end_off; j++)
                {
                    col = off_proc->idx2[j];
                    col_state = off_proc_states[col];
                    if (col_state == 1)
                    {
                        local_states[row] = 0;
                        break;
                    }

                    if (col_state == -1)
                    {
                        col_rand = off_proc_rands[col];
                        if (col_rand > row_rand)
                        {
                            break;
                        }
                        else if (col_rand == row_rand && col > row)
                        {
                            break;
                        }
                    }
                }

                if (j == row_end_off)
                {
                    make_coarse = true;
                }
            }


            if (make_coarse)
            {
                for (j = row_start_on; j < row_end_on; j++)
                {
                    col = on_proc->idx2[j];
                    if (local_states[col] == -1)
                    {
                        local_states[col] = 0;
                    }
                }
                for (j = row_start_off; j < row_end_off; j++)
                {
                    col = off_proc->idx2[j];
                    if (off_proc_states[col] == -1)
                    {
                        off_proc_states[col] = 0;
                    }
                }
                num_coarse++;
                local_states[row] = 1;
            }
            else
            {
                active = true;
            }
        }
        
        if (active)
        {
            for (int i = 0; i < comm->send_data->num_msgs; i++)
            {
                proc = comm->send_data->procs[i];
                start = comm->send_data->indptr[i];
                end = comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    int idx = comm->send_data->indices[j];
                    buffer[j] = local_states[idx];
                }
                MPI_Isend(&(buffer[start]), end - start, MPI_INT, proc, tag,
                        MPI_COMM_WORLD, &(comm->send_data->requests[i]));
            }

            for (int i = 0; i < comm->recv_data->num_msgs; i++)
            {
                proc = comm->recv_data->procs[i];
                start = comm->recv_data->indptr[i];
                end = comm->recv_data->indptr[i+1];
                MPI_Irecv(&(off_proc_states[start]), end - start, MPI_INT, proc, tag,
                        MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
            }

            if (comm->send_data->num_msgs)
            {
                MPI_Waitall(comm->send_data->num_msgs, 
                        comm->send_data->requests, MPI_STATUSES_IGNORE);
            }

            if (comm->recv_data->num_msgs)
            {
                MPI_Waitall(comm->recv_data->num_msgs, 
                        comm->recv_data->requests, MPI_STATUSES_IGNORE);
            }
        }
    }

    // No more active, so no more updates
    // Only probe for recvs while waiting for ibarrier to be hit by all
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            for (int i = 0; i < comm->send_data->num_msgs; i++)
            {
                proc = comm->send_data->procs[i];
                start = comm->send_data->indptr[i];
                end = comm->send_data->indptr[i+1];
                for (int j = start; j < end; j++)
                {
                    int idx = comm->send_data->indices[j];
                    buffer[j] = local_states[idx];
                }
                MPI_Isend(&(buffer[start]), end - start, MPI_INT, proc, tag,
                        MPI_COMM_WORLD, &(comm->send_data->requests[i]));
            }

            for (int i = 0; i < comm->recv_data->num_msgs; i++)
            {
                proc = comm->recv_data->procs[i];
                start = comm->recv_data->indptr[i];
                end = comm->recv_data->indptr[i+1];
                MPI_Irecv(&(off_proc_states[start]), end - start, MPI_INT, proc, tag,
                        MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
            }

            if (comm->send_data->num_msgs)
            {
                MPI_Waitall(comm->send_data->num_msgs, 
                        comm->send_data->requests, MPI_STATUSES_IGNORE);
            }

            if (comm->recv_data->num_msgs)
            {
                MPI_Waitall(comm->recv_data->num_msgs, 
                        comm->recv_data->requests, MPI_STATUSES_IGNORE);
            }
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }

    return num_coarse;

}


ParCSRMatrix* ParCSRMatrix::aggregate()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int j, ctr;
    int start, end;
    int local_col, proc;
    int global_col, proc_first_local_col;
    int col, col_T;
    int tag = 9420;
    std::vector<int> local_states;
    std::vector<int> off_proc_states;
    std::vector<int> T_proc_sizes(num_procs);
    std::vector<int> T_proc_starts(num_procs+1);

    ParCSRMatrix* S_sq = this->mult(this);
    S_sq->comm = new ParComm(S_sq->off_proc_column_map, S_sq->first_local_row,
            S_sq->first_local_col, S_sq->global_num_cols, S_sq->local_num_cols);
    int local_num_cols = S_sq->maximal_independent_set(local_states, off_proc_states);
    delete S_sq;

    MPI_Allgather(&local_num_cols, 1, MPI_INT, T_proc_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);

    T_proc_starts[0] = 0;
    int first_local_col = 0;
    for (int i = 0; i < rank; i++)
    {
        first_local_col += T_proc_sizes[i];
        T_proc_starts[i+1] = first_local_col;
    }
    int global_num_cols = first_local_col + local_num_cols;
    for (int i = rank + 1; i < num_procs; i++)
    {
        T_proc_starts[i] = global_num_cols;
        global_num_cols += T_proc_sizes[i];
    }

    ParCSRMatrix* T = new ParCSRMatrix(global_num_rows, global_num_cols,
        local_num_rows, local_num_cols, first_local_row, first_local_col);

    // Go through local states, changing all 0 (fine points) to -1, and
    // incrementing all coarse points (0-local_num_coarse), representing
    // local column indices
    ctr = 0;
    for (std::vector<int>::iterator it = local_states.begin(); it != local_states.end();
            ++it)
    {
        if (*it)
        {
            *it = first_local_col + ctr++;
        }
        else
        {
            *it = -1;
        }
    }

    // Communicate updated local states and recv from off_proc
    // to form off_proc_column_map
    std::vector<int> buffer;
    if (comm->send_data->size_msgs)
    {
        buffer.resize(comm->send_data->size_msgs);
    }
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (j = start; j < end; j++)
        {
            int idx = comm->send_data->indices[j];
            buffer[j] = local_states[idx];
        }
        MPI_Isend(&(buffer[start]), end - start, MPI_INT, proc, tag,
                MPI_COMM_WORLD, &(comm->send_data->requests[i]));
    }

    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        MPI_Irecv(&(off_proc_states[start]), end - start, MPI_INT, proc, tag,
                MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
    }

    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, 
                comm->send_data->requests, MPI_STATUSES_IGNORE);
    }

    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs,
                comm->recv_data->requests, MPI_STATUSES_IGNORE);
    }    
    
    // Go through each row of S... if local_states[i] >= 0, coarse_col[i] =
    // local_states[i] and continue
    // Otherwise, if a col in i such that local_states[col] >= 0, change
    // coarse_col[i] = local_states[col] and break
    //
    // Change local_states[i] to coarse_col[i], for all i
    // Go through S again... local_states[i] == -1, can add to any
    // neighbor aggregate (find local_states[col] >= 0 and make
    // coarse_col[i] = local_states[col]) and break
    //
    // Lastly, vertex with no neighbors (only the diagonal)... we will leave
    // this with local_states[i] == -1... becomes a 0 row in T
    std::vector<int> coarse_col;
    if (T->local_num_rows)
    {
        coarse_col.resize(T->local_num_rows);
    }
    for (int i = 0; i < T->local_num_rows; i++)
    {
        if (local_states[i] >= 0)
        {
            coarse_col[i] = local_states[i];
            continue;
        }
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (j = start; j < end; j++)
        {
            col = on_proc->idx2[j];
            if (local_states[col] >= 0)
            {
                coarse_col[i] = local_states[col];
                break;
            }
        }
        if (j == end)
        {
            start = off_proc->idx1[i];
            end = off_proc->idx1[i+1];
            for (j = start; j < end; j++)
            {
                col = off_proc->idx2[j];
                if (off_proc_states[col] >= 0)
                {
                    coarse_col[i] = off_proc_states[col];
                    break;
                }
            }
        }
    }

    for (int i = 0; i < T->local_num_rows; i++)
    {
        local_states[i] = coarse_col[i];
    }

    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        for (j = start; j < end; j++)
        {
            int idx = comm->send_data->indices[j];
            buffer[j] = local_states[idx];
        }
        MPI_Isend(&(buffer[start]), end - start, MPI_INT, proc, tag,
                MPI_COMM_WORLD, &(comm->send_data->requests[i]));
    }

    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        MPI_Irecv(&(off_proc_states[start]), end - start, MPI_INT, proc, tag,
                MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
    }

    if (comm->send_data->num_msgs)
    {
        MPI_Waitall(comm->send_data->num_msgs, 
                comm->send_data->requests, MPI_STATUSES_IGNORE);
    }

    if (comm->recv_data->num_msgs)
    {
        MPI_Waitall(comm->recv_data->num_msgs,
                comm->recv_data->requests, MPI_STATUSES_IGNORE);
    }  


    for (int i = 0; i < T->local_num_rows; i++)
    {
        if (local_states[i] >= 0)
        {
            continue;
        }
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (j = start; j < end; j++)
        {
            col = on_proc->idx2[j];
            if (local_states[col] >= 0)
            {
                coarse_col[i] = local_states[col];
                break;
            }
        }
        if (j == end)
        {
            start = off_proc->idx1[i];
            end = off_proc->idx1[i+1];
            for (j = start; j < end; j++)
            {
                col = off_proc->idx2[j];
                if (off_proc_states[col] >= 0)
                {
                    coarse_col[i] = off_proc_states[col];
                    break;
                }
            }
        }
    }

    // Create tentative interpolation operator based on 
    // distance 2 maximal independent set (each node is connected
    // to exactly one coarse point -- or is coarse point itself)
    T->on_proc->idx1[0] = 0;
    T->off_proc->idx1[0] = 0;
    int last_local_col = T->first_local_col + T->local_num_cols;
    std::set<int> global_col_set;
    std::map<int, int> global_to_local;
    for (int i = 0; i < T->local_num_rows; i++)
    {
        global_col = coarse_col[i];
        if (global_col >= 0)
        {
            if (global_col >= T->first_local_col && global_col < last_local_col)
            {
                T->on_proc->idx2.push_back(global_col - T->first_local_col);
                T->on_proc->vals.push_back(1.0);
            }
            else
            {
                T->off_proc->idx2.push_back(global_col);
                T->off_proc->vals.push_back(1.0);
            }
        }
        T->on_proc->idx1[i+1] = T->on_proc->idx2.size();
        T->off_proc->idx1[i+1] = T->off_proc->idx2.size();
    }
    T->on_proc->nnz = T->on_proc->idx2.size();
    T->off_proc->nnz = T->off_proc->idx2.size();

    T->off_proc->condense_cols();
    T->off_proc_column_map = T->off_proc->get_col_list();
    T->off_proc_num_cols = T->off_proc_column_map.size();

    return T;
}

