// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "repartition.hpp"

void make_contiguous(ParCSRMatrix* A)
{
    int rank;
    int num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    
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
    RAPtor_MPI_Request barrier_request;
    

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
    RAPtor_MPI_Allgather(&(A->on_proc_num_cols), 1, RAPtor_MPI_INT, proc_num_cols.data(), 1, RAPtor_MPI_INT,
            RAPtor_MPI_COMM_WORLD);

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
            RAPtor_MPI_Issend(&(send_buffer[start]), (end - start), RAPtor_MPI_INT, proc, 
                    send_key, RAPtor_MPI_COMM_WORLD, &(send_requests[n_sent++]));
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

    RAPtor_MPI_Status recv_status;
    while (size_recvs < local_assumed_num_cols)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        proc = recv_status.MPI_SOURCE;
        int recvbuf[count];
        RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, &recv_status);
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
        RAPtor_MPI_Waitall(n_sent, send_requests.data(), RAPtor_MPI_STATUS_IGNORE);
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
            RAPtor_MPI_Issend(&(send_buffer[start]), end - start, RAPtor_MPI_INT, proc, send_key, 
                    RAPtor_MPI_COMM_WORLD, &(send_requests[n_sent]));
            RAPtor_MPI_Irecv(&(recv_buffer[start]), end - start, RAPtor_MPI_INT, proc, recv_key, 
                    RAPtor_MPI_COMM_WORLD, &(recv_requests[n_sent++]));
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
        RAPtor_MPI_Testall(n_sent, send_requests.data(), &finished, RAPtor_MPI_STATUSES_IGNORE);
        while (!finished)
        {
            RAPtor_MPI_Iprobe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &msg_avail, &recv_status);
            if (msg_avail)
            {
                RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
                proc = recv_status.MPI_SOURCE;
                int recvbuf[count];
                RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, 
                        &recv_status);
                for (int i = 0; i < count; i++)
                {
                    orig_col = recvbuf[i];
                    assumed_col = orig_col - assumed_first_col;
                    new_col = assumed_col_to_new[assumed_col];
                    recvbuf[i] = new_col;
                }
                RAPtor_MPI_Send(recvbuf, count, RAPtor_MPI_INT, proc, recv_key, RAPtor_MPI_COMM_WORLD);
            }
            RAPtor_MPI_Testall(n_sent, send_requests.data(), &finished, RAPtor_MPI_STATUSES_IGNORE);
        }
    }
    RAPtor_MPI_Ibarrier(RAPtor_MPI_COMM_WORLD, &barrier_request);
    RAPtor_MPI_Test(&barrier_request, &finished, RAPtor_MPI_STATUS_IGNORE);
    while (!finished)
    {
        RAPtor_MPI_Iprobe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
            proc = recv_status.MPI_SOURCE;
            int recvbuf[count];
            RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, 
                    &recv_status);
            for (int i = 0; i < count; i++)
            {
                orig_col = recvbuf[i];
                assumed_col = orig_col - assumed_first_col;
                new_col = assumed_col_to_new[assumed_col];
                recvbuf[i] = new_col;
            }
            RAPtor_MPI_Send(recvbuf, count, RAPtor_MPI_INT, proc, recv_key, RAPtor_MPI_COMM_WORLD);
        }
        RAPtor_MPI_Test(&barrier_request, &finished, RAPtor_MPI_STATUS_IGNORE);
    }

    // Wait for recvs to complete, and map original local cols to new global
    // columns
    if (n_sent)
    {
        RAPtor_MPI_Waitall(n_sent, recv_requests.data(), RAPtor_MPI_STATUSES_IGNORE);
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



// Send rows to process in partition (with global row / global col)
ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, aligned_vector<int>& new_local_rows)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    new_local_rows.clear();

    ParCSRMatrix* A_part;

    aligned_vector<int> proc_sizes(num_procs, 0);
    aligned_vector<int> send_procs;
    aligned_vector<int> send_ptr;
    aligned_vector<int> send_indices;
    aligned_vector<int> send_sizes;
    aligned_vector<MPI_Request> requests;
    aligned_vector<char> send_buffer;
    aligned_vector<char> recv_buffer;

    if (A->local_num_rows) 
        send_indices.resize(A->local_num_rows);

    int n_sends, n_recvs;
    int proc, proc_idx, idx, new_ctr, ctr;
    int int_bytes, double_bytes;
    int start, end, count;
    int row_start, row_end;
    int row_size, row, global_row, global_col;
    int key = 29412;
    double value;
    MPI_Status status;

    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        if (proc_sizes[proc] == 0)
            send_procs.push_back(proc);
        proc_sizes[proc]++;
    }
    n_sends = send_procs.size();
    send_ptr.resize(n_sends+1);
    send_sizes.resize(n_sends, 0);
    requests.resize(n_sends);
    send_ptr[0] = 0;
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        send_ptr[i+1] = send_ptr[i] + proc_sizes[proc];
        proc_sizes[proc] = i;
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_idx = proc_sizes[proc];
        idx = send_ptr[proc_idx] + send_sizes[proc_idx]++;
        send_indices[idx] = i;
    }

    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_bytes);
    MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &double_bytes);
    int n_bytes = (2 * A->local_num_rows * int_bytes)
        + (A->local_nnz * int_bytes)
        + (A->local_nnz * double_bytes);
    if (n_bytes) send_buffer.resize(n_bytes);
    ctr = 0;
    new_ctr = 0;
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = send_indices[j];
            global_row = A->local_row_map[row];
            MPI_Pack(&(global_row), 1, MPI_INT, send_buffer.data(), n_bytes, 
                    &ctr, MPI_COMM_WORLD);

            row_size = (A->on_proc->idx1[row+1] - A->on_proc->idx1[row]) + 
                (A->off_proc->idx1[row+1] - A->off_proc->idx1[row]);
            MPI_Pack(&(row_size), 1, MPI_INT, send_buffer.data(), n_bytes, 
                    &ctr, MPI_COMM_WORLD);

            row_start = A->on_proc->idx1[row];
            row_end = A->on_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                MPI_Pack(&(A->on_proc_column_map[A->on_proc->idx2[k]]), 1, MPI_INT,
                        send_buffer.data(), n_bytes, &ctr, MPI_COMM_WORLD);
                MPI_Pack(&(A->on_proc->vals[k]), 1, MPI_DOUBLE, send_buffer.data(),
                        n_bytes, &ctr, MPI_COMM_WORLD);
            }
            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                MPI_Pack(&(A->off_proc_column_map[A->off_proc->idx2[k]]), 1, MPI_INT,
                        send_buffer.data(), n_bytes, &ctr, MPI_COMM_WORLD);
                MPI_Pack(&(A->off_proc->vals[k]), 1, MPI_DOUBLE, send_buffer.data(),
                        n_bytes, &ctr, MPI_COMM_WORLD);
            }
        }
        MPI_Isend(&(send_buffer[new_ctr]), ctr - new_ctr, MPI_PACKED, proc, 
                key, MPI_COMM_WORLD, &(requests[i]));
        new_ctr = ctr;
    }

    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        proc_sizes[proc] = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, proc_sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    n_recvs = proc_sizes[rank];
    CSRMatrix* recv_mat = new CSRMatrix();
    recv_mat->idx1.push_back(0);
    for (int i = 0; i < n_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, key, MPI_COMM_WORLD, &status);
        proc = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_PACKED, &count);
        if (count > recv_buffer.size())
            recv_buffer.resize(count);
        MPI_Recv(recv_buffer.data(), count, MPI_PACKED, proc, key, 
                MPI_COMM_WORLD, &status);

        ctr = 0;
        while (ctr < count)
        {
            MPI_Unpack(recv_buffer.data(), count, &ctr, &global_row, 1, MPI_INT, MPI_COMM_WORLD);
            new_local_rows.push_back(global_row);
            MPI_Unpack(recv_buffer.data(), count, &ctr, &row_size, 1, MPI_INT, MPI_COMM_WORLD);
            for (int j = 0; j < row_size; j++)
            {
                MPI_Unpack(recv_buffer.data(), count, &ctr, &global_col, 1, MPI_INT, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer.data(), count, &ctr, &value, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                recv_mat->idx2.push_back(global_col);
                recv_mat->vals.push_back(value);
            }
            recv_mat->idx1.push_back(recv_mat->idx2.size());
        }
    }
    if (n_sends) MPI_Waitall(n_sends, requests.data(), MPI_STATUSES_IGNORE);
    recv_mat->n_rows = recv_mat->idx1.size() - 1;
    recv_mat->nnz = recv_mat->idx2.size();

    MPI_Allgather(&(recv_mat->n_rows), 1, MPI_INT, proc_sizes.data(), 1, 
            MPI_INT, MPI_COMM_WORLD);
    int first_local_row = 0;
    for (int i = 0; i < rank; i++)
        first_local_row += proc_sizes[i];
    A_part = new ParCSRMatrix(A->global_num_rows, A->global_num_rows, 
            recv_mat->n_rows, recv_mat->n_rows,
            first_local_row, first_local_row, A->partition->topology);

    std::map<int, int> local_row_map;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        global_row = first_local_row + i;
        local_row_map[global_row] = i;
        A_part->on_proc_column_map.push_back(global_row);
    }
    A_part->local_row_map = A_part->get_on_proc_column_map();
    A_part->on_proc_num_cols = A_part->on_proc_column_map.size();
    
    ctr = 0;
    std::map<int, int> off_proc_col_map;
    std::map<int, int>::iterator it;
    for (int i = 0; i < A_part->local_num_rows; i++)
    {
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = recv_mat->idx2[j];
            it = local_row_map.find(global_col);
            if (it == local_row_map.end())
            {
                it = off_proc_col_map.find(global_col);
                if (it == off_proc_col_map.end())
                {
                    off_proc_col_map[global_col] = ctr;
                    A_part->off_proc->idx2.push_back(ctr++);
                    A_part->off_proc_column_map.push_back(global_col);
                }
                else
                {
                    A_part->off_proc->idx2.push_back(it->second);
                }
                A_part->off_proc->vals.push_back(recv_mat->vals[j]);
            }
            else
            {
                A_part->on_proc->idx2.push_back(it->second);
                A_part->on_proc->vals.push_back(recv_mat->vals[j]);
            }
        }
        A_part->on_proc->idx1[i+1] = A_part->on_proc->idx2.size();
        A_part->off_proc->idx1[i+1] = A_part->off_proc->idx2.size();
    }
    A_part->on_proc->nnz = A_part->on_proc->idx2.size();
    A_part->off_proc->nnz = A_part->off_proc->idx2.size();
    A_part->local_nnz = A_part->on_proc->nnz + A_part->off_proc->nnz;
    A_part->off_proc_num_cols = ctr;

    delete recv_mat;

    key++;
    aligned_vector<int> assumed_row_procs(A_part->local_num_rows);
    std::fill(proc_sizes.begin(), proc_sizes.end(), 0);
    send_procs.clear();
    int assumed_local_n = A->global_num_rows / num_procs;
    int extra = A->global_num_rows % num_procs;
    int first_extra = extra * (assumed_local_n+1);
    int first_assumed_row = rank * assumed_local_n;
    if (rank < extra) first_assumed_row += rank;
    else first_assumed_row += extra;

    for (int i = 0; i < A_part->local_num_rows; i++)
    {
        global_row = new_local_rows[i];
        if (global_row < first_extra)
        {
            proc = global_row / (assumed_local_n + 1);
        }
        else
        {
            proc = ((global_row - first_extra) / assumed_local_n) + extra;
        }
        assumed_row_procs[i] = proc;
        if (proc_sizes[proc] == 0)
        {
            send_procs.push_back(proc);
        }
        proc_sizes[proc]++;
    }
    n_sends = send_procs.size();
    send_ptr.resize(n_sends+1);
    requests.resize(n_sends);
    send_sizes.resize(n_sends);
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        send_ptr[i+1] = send_ptr[i] + proc_sizes[proc];
        proc_sizes[proc] = i;
        send_sizes[i] = 0;
    }
    send_indices.resize(A_part->local_num_rows);
    for (int i = 0; i < A_part->local_num_rows; i++)
    {
        proc = assumed_row_procs[i];
        proc_idx = proc_sizes[proc];
        idx = send_ptr[proc_idx] + send_sizes[proc_idx]++;
        send_indices[idx] = i;
    }

    aligned_vector<int> sendbuf(2*A_part->local_num_rows);
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = send_indices[j];
            sendbuf[2*j] = new_local_rows[row];
            sendbuf[2*j+1] = A_part->local_row_map[row];
        }
        MPI_Isend(&(sendbuf[2*start]), 2*(end-start), MPI_INT, proc, key,
                MPI_COMM_WORLD, &(requests[i]));
        proc_sizes[proc] = 1;
    }
    
    aligned_vector<int> assumed_row_indices(assumed_local_n + 1);
    aligned_vector<int> recvbuf;
    MPI_Allreduce(MPI_IN_PLACE, proc_sizes.data(), num_procs, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
    n_recvs = proc_sizes[rank];
    for (int i = 0; i < n_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, key, MPI_COMM_WORLD, &status);
        proc = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_INT, &count);
        if (count > recvbuf.size())
            recvbuf.resize(count);
        MPI_Recv(recvbuf.data(), count, MPI_INT, proc, key, 
                MPI_COMM_WORLD, &status);
        for (int j = 0; j < count; j += 2)
        {
            int orig_row = recvbuf[j];
            int new_row = recvbuf[j+1];
            int local_row = orig_row - first_assumed_row;
            assumed_row_indices[local_row] = new_row;
        }
    }

    if (n_sends) MPI_Waitall(n_sends, requests.data(), MPI_STATUSES_IGNORE);



    // Now, send off_proc_cols (global col) to assumed proc 
    key++;
    send_procs.clear();
    std::fill(proc_sizes.begin(), proc_sizes.end(), 0);
    assumed_row_procs.resize(A_part->off_proc_num_cols);
    for (int i = 0; i < A_part->off_proc_num_cols; i++)
    {
        global_row = A_part->off_proc_column_map[i];
        if (global_row < first_extra)
        {
            proc = global_row / (assumed_local_n + 1);
        }
        else
        {
            proc = ((global_row - first_extra) / assumed_local_n) + extra;
        }
        assumed_row_procs[i] = proc;
        if (proc_sizes[proc] == 0)
        {
            send_procs.push_back(proc);
        }
        proc_sizes[proc]++;
    }
    n_sends = send_procs.size();
    send_ptr.resize(n_sends+1);
    requests.resize(n_sends);
    send_sizes.resize(n_sends);
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        send_ptr[i+1] = send_ptr[i] + proc_sizes[proc];
        proc_sizes[proc] = i;
        send_sizes[i] = 0;
    }
    send_indices.resize(A_part->off_proc_num_cols);
    for (int i = 0; i < A_part->off_proc_num_cols; i++)
    {
        proc = assumed_row_procs[i];
        proc_idx = proc_sizes[proc];
        idx = send_ptr[proc_idx] + send_sizes[proc_idx]++;
        send_indices[idx] = i;
    }

    sendbuf.resize(A_part->off_proc_num_cols);
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = send_indices[j];
            sendbuf[j] = A_part->off_proc_column_map[row];
        }
        MPI_Isend(&(sendbuf[start]), (end-start), MPI_INT, proc, key,
                MPI_COMM_WORLD, &(requests[i]));
        proc_sizes[proc] = 1;
    }
    
    MPI_Allreduce(MPI_IN_PLACE, proc_sizes.data(), num_procs, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
    n_recvs = proc_sizes[rank];
    aligned_vector<MPI_Request> return_requests;
    if (n_recvs) return_requests.resize(n_recvs);
    for (int i = 0; i < n_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, key, MPI_COMM_WORLD, &status);
        proc = status.MPI_SOURCE;
        MPI_Get_count(&status, MPI_INT, &count);
        if (count > recvbuf.size())
            recvbuf.resize(count);
        MPI_Recv(recvbuf.data(), count, MPI_INT, proc, key, 
                MPI_COMM_WORLD, &status);
        for (int j = 0; j < count; j++)
        {
            int orig_row = recvbuf[j];
            int local_row = orig_row - first_assumed_row;
            int new_row = assumed_row_indices[local_row];
            recvbuf[j] = new_row;
        }
        MPI_Isend(recvbuf.data(), count, MPI_INT, proc, key+1, 
                MPI_COMM_WORLD, &(return_requests[i]));
    }

    if (n_sends) 
    {
        MPI_Waitall(n_sends, requests.data(), MPI_STATUSES_IGNORE);
        for (int i = 0; i < n_sends; i++)
        {
            proc = send_procs[i];
            start = send_ptr[i];
            end = send_ptr[i+1];
            MPI_Irecv(&(sendbuf[start]), end - start, MPI_INT, proc, key+1,
                    MPI_COMM_WORLD, &(requests[i]));
        }
        MPI_Waitall(n_sends, requests.data(), MPI_STATUSES_IGNORE);
    }
    for (int i = 0; i < n_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        for (int j = start; j < end; j++)
        {
            row = send_indices[j];
            A_part->off_proc_column_map[row] = sendbuf[row];            
        }
    }

    aligned_vector<int> off_proc_cols(A_part->off_proc_num_cols);
    aligned_vector<int> off_proc_new_cols(A_part->off_proc_num_cols);
    std::iota(off_proc_cols.begin(), off_proc_cols.end(), 0);
    vec_sort(A_part->off_proc_column_map, off_proc_cols);
    for (int i = 0; i < A_part->off_proc_num_cols; i++)
        off_proc_new_cols[off_proc_cols[i]] = i;
    for (int i = 0; i < A_part->off_proc->nnz; i++)
    {
        idx = A_part->off_proc->idx2[i];
        A_part->off_proc->idx2[i] = off_proc_new_cols[idx];
    }

    A_part->comm = new ParComm(A_part->partition, A_part->off_proc_column_map, 
            A_part->on_proc_column_map);

    // Sort rows, removing duplicate entries and moving diagonal 
    // value to first
    A->on_proc->sort();
    A->on_proc->move_diag();
    A->off_proc->sort();

    return A_part;
}

