// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "repartition.hpp"

void make_contiguous(ParCSRMatrix* A)
{
    int rank;
    int num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    
    int assumed_num_cols, assumed_first_col, assumed_last_col;
    int local_assumed_num_cols;
    int assumed_proc, orig_col, new_col;
    int proc, proc_idx, idx, ctr;
    int num_sends, n_sent, send_size;
    int size_recvs;
    int send_key, recv_key;
    
    int start, end;
    int assumed_col, local_col;
    int msg_avail, finished, count;
    MPI_Request barrier_request;
    

    aligned_vector<int> assumed_col_to_new;

    aligned_vector<int> proc_num_cols(num_procs);
    aligned_vector<int> send_procs;
    aligned_vector<int> send_ptr;
    aligned_vector<int> send_ctr;
    aligned_vector<int> send_buffer;
    aligned_vector<int> recv_buffer;
    aligned_vector<MPI_Request> send_requests;

    aligned_vector<MPI_Request> recv_requests;

    std::map<int, int> global_to_local;
    ctr = 0;
    for (aligned_vector<int>::const_iterator it = A->off_proc_column_map.begin();  
            it != A->off_proc_column_map.end(); ++it)
    {
        global_to_local[*it] = ctr++;
    }

    // Find how many columns are local to each process
    MPI_Allgather(&(A->on_proc_num_cols), 1, MPI_INT, proc_num_cols.data(), 1, MPI_INT,
            MPI_COMM_WORLD);

    // Determine the new first local row / first local col of rank
    A->partition->first_local_col = 0;
    for (int i = 0; i < rank; i++)
    {
        A->partition->first_local_col += proc_num_cols[i];
    }
    A->partition->first_local_row = A->partition->first_local_col;

    // Determine the global number of columns and rows
    A->global_num_cols = A->partition->first_local_col;
    for (int i = rank; i < num_procs; i++)
    {
        A->global_num_cols += proc_num_cols[i];
    }
    A->global_num_rows = A->global_num_cols;

    // Determine which columns are assumed local to rank
    assumed_num_cols = ((A->global_num_cols - 1) / num_procs) + 1;
    assumed_first_col = assumed_num_cols * rank;
    assumed_last_col = assumed_first_col + assumed_num_cols;
    if (assumed_first_col > A->global_num_cols)
        assumed_first_col = A->global_num_cols;
    if (assumed_last_col > A->global_num_cols)
        assumed_last_col = A->global_num_cols;
    local_assumed_num_cols = assumed_last_col - assumed_first_col;

    if (local_assumed_num_cols)
    {
        assumed_col_to_new.resize(local_assumed_num_cols);
    }

    // Determine number of local cols assumed to be on each distant proc
    for (int i = 0; i < num_procs; i++)
    {
        proc_num_cols[i] = 0;
    }
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        orig_col = A->on_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        proc_num_cols[assumed_proc]++;
    }

    // Re-arrange on_proc cols, ordered by assumed proc
    send_size = 0;
    send_ptr.emplace_back(send_size);
    for (int i = 0; i < num_procs; i++)
    {
        if (proc_num_cols[i])
        {
            send_size += proc_num_cols[i];
            proc_num_cols[i] = send_procs.size();
            send_procs.emplace_back(i);
            send_ptr.emplace_back(send_size);
        }
    }
    num_sends = send_procs.size();
    if (send_size)
    {
        send_ctr.resize(num_sends, 0);
        send_buffer.resize(2*send_size);
        send_requests.resize(num_sends);
        for (int i = 0; i < A->on_proc_num_cols; i++)
        {
            orig_col = A->on_proc_column_map[i];
            new_col = A->partition->first_local_col + i;
            assumed_proc = orig_col / assumed_num_cols;
            proc_idx = proc_num_cols[assumed_proc];
            idx = send_ptr[proc_idx] + send_ctr[proc_idx]++;
            send_buffer[2*idx] = orig_col;
            send_buffer[2*idx+1] = new_col;
        }
    }

    n_sent = 0;
    size_recvs = 0;
    send_key = 7568;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = 2*send_ptr[i];
        end = 2*send_ptr[i+1];
        if (proc != rank)
        {
            MPI_Issend(&(send_buffer[start]), (end - start), MPI_INT, proc, 
                    send_key, MPI_COMM_WORLD, &(send_requests[n_sent++]));
        }
        else
        {
            for (int j = start; j < end; j+=2)
            {
                orig_col = send_buffer[j];
                new_col = send_buffer[j+1];
                local_col = orig_col - assumed_first_col;
                assumed_col_to_new[local_col] = new_col;
            }
            size_recvs += ((end - start) / 2);
        }
    }

    MPI_Status recv_status;
    while (size_recvs < local_assumed_num_cols)
    {
        MPI_Probe(MPI_ANY_SOURCE, send_key, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, MPI_INT, &count);
        proc = recv_status.MPI_SOURCE;
        int recvbuf[count];
        MPI_Recv(recvbuf, count, MPI_INT, proc, send_key, MPI_COMM_WORLD, &recv_status);
        for (int i = 0; i < count; i+= 2)
        {
            orig_col = recvbuf[i];
            new_col = recvbuf[i+1];
            local_col = orig_col - assumed_first_col;
            assumed_col_to_new[local_col] = new_col;
        }
        size_recvs += (count / 2);
    }

    if (n_sent)
    {
        MPI_Waitall(n_sent, send_requests.data(), MPI_STATUS_IGNORE);
    }


    // Reset proc_num_cols values to 0
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        proc_num_cols[proc] = 0;
    }

    // Clear send info from previous communication
    send_procs.clear();
    send_ptr.clear();
    send_ctr.clear();

    // Go through off_proc columns, and find which proc with which each is
    // assumed to be associated
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        orig_col = A->off_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        proc_num_cols[assumed_proc]++;
    }

    // Create send_procs, send_ptr
    send_size = 0;
    send_ptr.emplace_back(send_size);
    for (int i = 0; i < num_procs; i++)
    {
        if (proc_num_cols[i])
        {
            send_size += proc_num_cols[i];
            proc_num_cols[i] = send_procs.size();
            send_procs.emplace_back(i);
            send_ptr.emplace_back(send_size);
        }
    }
    num_sends = send_procs.size();
    if (num_sends)
    {
        send_ctr.resize(num_sends, 0);
        send_buffer.resize(send_size);
        recv_buffer.resize(send_size);
        send_requests.resize(num_sends);
        recv_requests.resize(num_sends);
    }
    
    // Add columns to send buffer, ordered by assumed process
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        orig_col = A->off_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        proc_idx = proc_num_cols[assumed_proc];
        idx = send_ptr[proc_idx] + send_ctr[proc_idx]++;
        send_buffer[idx] = orig_col;
    }

    // Send off_proc_columns to proc assumed to hold col and recv new global idx
    // of column.  If assumed proc is rank, find new col and add to
    // off_proc_col_to_new
    n_sent = 0;
    send_key = 7980;
    recv_key = 8976;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];

        if (proc != rank)
        {
            MPI_Issend(&(send_buffer[start]), end - start, MPI_INT, proc, send_key, 
                    MPI_COMM_WORLD, &(send_requests[n_sent]));
            MPI_Irecv(&(recv_buffer[start]), end - start, MPI_INT, proc, recv_key, 
                    MPI_COMM_WORLD, &(recv_requests[n_sent++]));
        }
        else
        {
            for (int j = start; j < end; j++)
            {
                orig_col = send_buffer[j];
                assumed_col = orig_col - assumed_first_col;
                new_col = assumed_col_to_new[assumed_col];
                local_col = global_to_local[orig_col];
                A->off_proc_column_map[local_col] = new_col;
            }
        }
    }

    // Recv columns corresponding to my assumed cols, and return their new cols
    if (n_sent)
    {
        MPI_Testall(n_sent, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, send_key, MPI_COMM_WORLD, &msg_avail, &recv_status);
            if (msg_avail)
            {
                MPI_Get_count(&recv_status, MPI_INT, &count);
                proc = recv_status.MPI_SOURCE;
                int recvbuf[count];
                MPI_Recv(recvbuf, count, MPI_INT, proc, send_key, MPI_COMM_WORLD, 
                        &recv_status);
                for (int i = 0; i < count; i++)
                {
                    orig_col = recvbuf[i];
                    assumed_col = orig_col - assumed_first_col;
                    new_col = assumed_col_to_new[assumed_col];
                    recvbuf[i] = new_col;
                }
                MPI_Send(recvbuf, count, MPI_INT, proc, recv_key, MPI_COMM_WORLD);
            }
            MPI_Testall(n_sent, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, send_key, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            MPI_Get_count(&recv_status, MPI_INT, &count);
            proc = recv_status.MPI_SOURCE;
            int recvbuf[count];
            MPI_Recv(recvbuf, count, MPI_INT, proc, send_key, MPI_COMM_WORLD, 
                    &recv_status);
            for (int i = 0; i < count; i++)
            {
                orig_col = recvbuf[i];
                assumed_col = orig_col - assumed_first_col;
                new_col = assumed_col_to_new[assumed_col];
                recvbuf[i] = new_col;
            }
            MPI_Send(recvbuf, count, MPI_INT, proc, recv_key, MPI_COMM_WORLD);
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }

    // Wait for recvs to complete, and map original local cols to new global
    // columns
    if (n_sent)
    {
        MPI_Waitall(n_sent, recv_requests.data(), MPI_STATUSES_IGNORE);
    }
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        if (proc != rank)
        {
            for (int j = start; j < end; j++)
            {
                orig_col = send_buffer[j];
                new_col = recv_buffer[j];
                local_col = global_to_local[orig_col];
                A->off_proc_column_map[local_col] = new_col;
            }
        }
    }

    // re-index columns of off_proc (ordered by new global columns)
    if (A->off_proc_num_cols)
    {
        // Find permutation of off_proc columns, sorted by global 
        // column indices in ascending order
        aligned_vector<int> p(A->off_proc_num_cols);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), 
                [&](int i, int j)
                {
                    return A->off_proc_column_map[i] < A->off_proc_column_map[j];
                });
    
        // Form off_proc_orig_to_new, mapping original off_proc local
        // column indices to new local column indices
        aligned_vector<int> off_proc_orig_to_new(A->off_proc_num_cols);
        for (int i = 0; i < A->off_proc_num_cols; i++)
        {
            off_proc_orig_to_new[p[i]] = i;
        }

        // Re-index columns of off_proc
        for (aligned_vector<int>::iterator it = A->off_proc->idx2.begin();
                it != A->off_proc->idx2.end(); ++it)
        {
            *it = off_proc_orig_to_new[*it];
        }

        // Sort off_proc_column_map based on permutation vector p
        aligned_vector<bool> done(A->off_proc_num_cols);
        for (int i = 0; i < A->off_proc_num_cols; i++)
        {
            if (done[i]) continue;

            done[i] = true;
            int prev_j = i;
            int j = p[i];
            while (i != j)
            {
                std::swap(A->off_proc_column_map[prev_j], A->off_proc_column_map[j]);
                done[j] = true;
                prev_j = j;
                j = p[j];
            }
        }
    }

    for (int i = 0; i < A->local_num_rows; i++)
    {
        A->on_proc_column_map[i] = A->partition->first_local_col + i;
    }
    A->local_row_map = A->get_on_proc_column_map();

    A->comm = new ParComm(A->partition, A->off_proc_column_map);

    // Sort rows, removing duplicate entries and moving diagonal 
    // value to first
    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();
}

ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, aligned_vector<int>& new_local_rows)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParCSRMatrix* A_part;
    aligned_vector<int> send_row_buffer;
    aligned_vector<int> recv_row_buffer;
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    
    int proc, proc_idx;
    int idx, ctr;
    int num_rows, first_row;
    int start, end, col;
    int row_size;
    int send_row_size;
    int num_sends;
    int row_key = 370284;
    int key = 204853;
    int msg_avail, finished;
    int count;
    double val;
    MPI_Status recv_status;
    MPI_Request barrier_request;
    aligned_vector<int> send_procs;
    aligned_vector<int> send_ptr;
    aligned_vector<int> proc_sizes(num_procs, 0);
    aligned_vector<int> proc_to_idx(num_procs);
    aligned_vector<MPI_Request> send_requests;
    aligned_vector<MPI_Request> recv_requests;
    aligned_vector<int> recv_rows;
    aligned_vector<int> recv_row_sizes;
    aligned_vector<int> recv_procs;
    aligned_vector<int> recv_ptr;

    // Find how many rows go to each proc
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_sizes[proc]++;
    }

    // Create send_procs: procs to which I must send rows
    // and send_ptr: number of rows to send to each
    send_row_size = 0;
    send_ptr.emplace_back(0);
    for (int i = 0; i < num_procs; i++)
    {
        if (proc_sizes[i])
        {
            send_row_size += proc_sizes[i];
            proc_to_idx[i] = send_procs.size();
            send_procs.emplace_back(i);
            send_ptr.emplace_back(send_row_size);
            proc_sizes[i] = 0;
        }
    }

    // Now know the number of messages to be sent
    num_sends = send_procs.size();
    if (num_sends)
    {
        send_requests.resize(num_sends);
    }

    // Add row and row_size to send buffer, ordered by 
    // processes to which buffer is sent
    send_row_buffer.resize(2*send_row_size);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_idx = proc_to_idx[proc];
        idx = send_ptr[proc_idx] + proc_sizes[proc]++;
        send_row_buffer[2*idx] = A->local_row_map[i];
        
        row_size = (A->on_proc->idx1[i+1] - A->on_proc->idx1[i]) + 
            (A->off_proc->idx1[i+1] - A->off_proc->idx1[i]);
        send_row_buffer[2*idx+1] = row_size;
    }

    // Send to proc p the rows and row sizes that will be sent next
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Issend(&(send_row_buffer[2*start]), 2*(end - start), MPI_INT, proc,
                row_key, MPI_COMM_WORLD, &send_requests[i]);
    }

    // Dynamically receive rows and corresponding sizes  
    int recv_size = 0;
    recv_ptr.emplace_back(0);
    if (num_sends)
    {
        MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, row_key, MPI_COMM_WORLD, &msg_avail, &recv_status);
            if (msg_avail)
            {
                proc = recv_status.MPI_SOURCE;
                MPI_Get_count(&recv_status, MPI_INT, &count);
                if (count > (int) recv_row_buffer.size())
                {
                    recv_row_buffer.resize(count);
                }
                MPI_Recv(recv_row_buffer.data(), count, MPI_INT, proc, row_key,
                        MPI_COMM_WORLD, &recv_status);
                for (int i = 0; i < count; i += 2)
                {
                    recv_rows.emplace_back(recv_row_buffer[i]);
                    recv_row_sizes.emplace_back(recv_row_buffer[i+1]);
                    recv_size += recv_row_buffer[i+1];
                }
                recv_procs.emplace_back(proc);
                recv_ptr.emplace_back(recv_size);
            }
            MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, row_key, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            proc = recv_status.MPI_SOURCE;
            MPI_Get_count(&recv_status, MPI_INT, &count);
            if (count > (int) recv_row_buffer.size())
            {
                recv_row_buffer.resize(count);
            }
            MPI_Recv(recv_row_buffer.data(), count, MPI_INT, proc, row_key,
                    MPI_COMM_WORLD, &recv_status);
            for (int i = 0; i < count; i += 2)
            {
                recv_rows.emplace_back(recv_row_buffer[i]);
                recv_row_sizes.emplace_back(recv_row_buffer[i+1]);
                recv_size += recv_row_buffer[i+1];
            }
            recv_procs.emplace_back(proc);
            recv_ptr.emplace_back(recv_size);
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }
    int num_recvs = recv_procs.size();
    num_rows = recv_rows.size();
    recv_requests.resize(num_recvs);
    recv_buffer.resize(recv_size);

    // Form vector of pair data (col indices and values) for sending to each
    // proc
    for (int i = 0; i < num_procs; i++)
    {
        proc_sizes[i] = 0;
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        row_size = (A->on_proc->idx1[i+1] - A->on_proc->idx1[i]) + 
            (A->off_proc->idx1[i+1] - A->off_proc->idx1[i]);
        proc_sizes[proc] += row_size;
    }
    int size_send = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        size_send += proc_sizes[proc];
        send_ptr[i+1] = send_ptr[i] + proc_sizes[proc];
        proc_sizes[proc] = 0;
    }
    send_buffer.resize(size_send);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_idx = proc_to_idx[proc];
        idx = send_ptr[proc_idx] + proc_sizes[proc];

        start = A->on_proc->idx1[i];
        end = A->on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            send_buffer[idx].val = A->on_proc->vals[j];
            send_buffer[idx].index = A->on_proc_column_map[A->on_proc->idx2[j]];
            idx++;
        }

        start = A->off_proc->idx1[i];
        end = A->off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            send_buffer[idx].val = A->off_proc->vals[j];
            send_buffer[idx].index = A->off_proc_column_map[A->off_proc->idx2[j]];
            idx++;
        }

        proc_sizes[proc] = idx - send_ptr[proc_idx];
    } 

    // Send send_buffer (PairData)
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        MPI_Isend(&send_buffer[start], end - start, MPI_DOUBLE_INT, proc, key,
                MPI_COMM_WORLD, &send_requests[i]);
    }

    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        start = recv_ptr[i];
        end = recv_ptr[i+1];
        MPI_Irecv(&recv_buffer[start], end - start, MPI_DOUBLE_INT, proc, key,
                MPI_COMM_WORLD, &recv_requests[i]);
    }
    
    MPI_Waitall(num_sends, send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(num_recvs, recv_requests.data(), MPI_STATUSES_IGNORE);


    // Assuming local num cols == num_rows (square)
    MPI_Allgather(&(num_rows), 1, MPI_INT, proc_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_sizes[i];
    }

    A_part = new ParCSRMatrix(A->global_num_rows, A->global_num_rows, num_rows, num_rows, 
            first_row, first_row, A->partition->topology);

    // Create row_ptr
    // Add values/indices to appropriate positions
    std::map<int, int> on_proc_to_local;
    for(int i = 0; i < num_rows; i++)
    {
       on_proc_to_local[recv_rows[i]] = i;
       A_part->on_proc_column_map.emplace_back(recv_rows[i]);
    }
    A_part->local_row_map = A_part->get_on_proc_column_map();
    A_part->on_proc_num_cols = A_part->on_proc_column_map.size();

    ctr = 0;
    A_part->on_proc->idx1[0] = 0;
    A_part->off_proc->idx1[0] = 0;
    for (int i = 0; i < num_rows; i++)
    {
        row_size = recv_row_sizes[i];
        for (int j = 0; j < row_size; j++)
        {
            col = recv_buffer[ctr].index;
            val = recv_buffer[ctr++].val;

            if (on_proc_to_local.find(col) != on_proc_to_local.end())
            {
                A_part->on_proc->idx2.emplace_back(on_proc_to_local[col]);
                A_part->on_proc->vals.emplace_back(val);
            }
            else
            {
                A_part->off_proc->idx2.emplace_back(col);
                A_part->off_proc->vals.emplace_back(val);
            }
        }
        A_part->on_proc->idx1[i+1] = A_part->on_proc->idx2.size();
        A_part->off_proc->idx1[i+1] = A_part->off_proc->idx2.size();
    }
    A_part->on_proc->nnz = A_part->on_proc->idx2.size();
    A_part->off_proc->nnz = A_part->off_proc->idx2.size();
    A_part->local_nnz = A_part->on_proc->nnz + A_part->off_proc->nnz;

    aligned_vector<int> off_proc_cols;
    std::copy(A_part->off_proc->idx2.begin(), A_part->off_proc->idx2.end(),
            std::back_inserter(off_proc_cols));
    std::sort(off_proc_cols.begin(), off_proc_cols.end());
    int prev_col = -1;
    std::map<int, int> global_to_local;
    for (aligned_vector<int>::iterator it = off_proc_cols.begin(); 
            it != off_proc_cols.end(); ++it)
    {
        if (*it != prev_col)
        {
            global_to_local[*it] = A_part->off_proc_column_map.size();
            A_part->off_proc_column_map.emplace_back(*it);
            *it = prev_col;
        }
    }
    A_part->off_proc_num_cols = A_part->off_proc_column_map.size();

    for (aligned_vector<int>::iterator it = A_part->off_proc->idx2.begin();
            it != A_part->off_proc->idx2.end(); ++it)
    {
        *it = global_to_local[*it];
    }

    new_local_rows.resize(A_part->on_proc_num_cols);
    std::copy(A_part->on_proc_column_map.begin(), A_part->on_proc_column_map.end(),
            new_local_rows.begin());
    make_contiguous(A_part);

    return A_part;
}


