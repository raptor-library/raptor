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
    int count;

    aligned_vector<int> assumed_col_to_new;
    aligned_vector<int> proc_num_cols(num_procs);
    aligned_vector<int> send_procs(num_procs);
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
    std::fill(proc_num_cols.begin(), proc_num_cols.end(), 0);
    num_sends = 0;
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        orig_col = A->on_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        if (proc_num_cols[assumed_proc] == 0)
        {
            send_procs[num_sends++] = assumed_proc;
        }
        proc_num_cols[assumed_proc]++;
    }
    send_ptr.resize(num_sends+1);
    send_ptr[0] = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        send_ptr[i+1] = send_ptr[i] + proc_num_cols[proc];
        proc_num_cols[proc] = i;
    }
    send_size = send_ptr[num_sends];

    // Re-arrange on_proc cols, ordered by assumed proc
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
            RAPtor_MPI_Isend(&(send_buffer[start]), (end - start), RAPtor_MPI_INT, proc, 
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
    aligned_vector<int> recvbuf;
    while (size_recvs < local_assumed_num_cols)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        proc = recv_status.MPI_SOURCE;
        if (count > recvbuf.size()) recvbuf.resize(count);
        RAPtor_MPI_Recv(recvbuf.data(), count, RAPtor_MPI_INT, proc, send_key, 
                RAPtor_MPI_COMM_WORLD, &recv_status);
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

    // Go through off_proc columns, and find which proc with which each is
    // assumed to be associated
    send_procs.resize(num_procs);
    num_sends = 0;
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        orig_col = A->off_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        if (proc_num_cols[assumed_proc] == 0)
        {
            send_procs[num_sends++] = assumed_proc;
        }
        proc_num_cols[assumed_proc]++;
    }
    send_ptr.resize(num_sends+1);
    send_size = 0;
    send_ptr[0] = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        send_ptr[i+1] = send_ptr[i] + proc_num_cols[proc];
        proc_num_cols[i] = proc;
    }
    send_size = send_ptr[num_sends];


    // Create send_procs, send_ptr
    if (num_sends)
    {
        send_ctr.resize(num_sends);
        std::fill(send_ctr.begin(), send_ctr.end(), 0);
        send_buffer.resize(send_size);
        recv_buffer.resize(send_size);
        send_requests.resize(num_sends);
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
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        proc_num_cols[proc] = 1;
    }
    proc_num_cols[rank] = 0;
    MPI_Allreduce(MPI_IN_PLACE, proc_num_cols.data(), num_procs, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    int num_recvs = proc_num_cols[rank];
    if (num_recvs)
    {
        recv_requests.resize(num_recvs);
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
        proc_num_cols[proc] = i;
        start = send_ptr[i];
        end = send_ptr[i+1];

        if (proc != rank)
        {
            RAPtor_MPI_Issend(&(send_buffer[start]), end - start, RAPtor_MPI_INT, proc, send_key, 
                    RAPtor_MPI_COMM_WORLD, &(send_requests[n_sent++]));
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
    for (int i = 0; i < num_recvs; i++)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, send_key, RAPtor_MPI_COMM_WORLD,
                &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        proc = recv_status.MPI_SOURCE;
        if (count > recvbuf.size()) recvbuf.resize(count);
        RAPtor_MPI_Recv(recvbuf.data(), count, RAPtor_MPI_INT, proc, send_key, RAPtor_MPI_COMM_WORLD, 
                &recv_status);
        for (int j = 0; j < count; j++)
        {
            orig_col = recvbuf[j];
            assumed_col = orig_col - assumed_first_col;
            new_col = assumed_col_to_new[assumed_col];
            recvbuf[j] = new_col;
        }
        RAPtor_MPI_Isend(recvbuf.data(), count, RAPtor_MPI_INT, proc, recv_key, 
                RAPtor_MPI_COMM_WORLD, &(recv_requests[i]));
    }

    // Wait for recvs to complete, and map original local cols to new global
    // columns
    if (n_sent)
    {
        RAPtor_MPI_Waitall(n_sent, send_requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    }

    n_sent = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        if (proc != rank)
        {
            RAPtor_MPI_Irecv(&(recv_buffer[start]), end - start, RAPtor_MPI_INT, proc,
                    recv_key, RAPtor_MPI_COMM_WORLD, &(send_requests[n_sent++]));
        }
    }
    if (num_recvs)
    {
        RAPtor_MPI_Waitall(num_recvs, recv_requests.data(), RAPtor_MPI_STATUSES_IGNORE);
    }
    if (n_sent)
    {
        RAPtor_MPI_Waitall(n_sent, send_requests.data(), RAPtor_MPI_STATUSES_IGNORE);
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


    ParCSRMatrix* A_part = NULL;

    int proc, start, end;
    int row_start, row_end, row_size;
    int num_sends, num_recvs;
    int proc_idx, idx, ctr, prev_ctr;
    int row, col, global_row, global_col;
    int count, first_row;
    double val;
    aligned_vector<int> proc_rows(num_procs, 0);
    aligned_vector<int> proc_to_idx(num_procs);
    aligned_vector<int> send_procs(num_procs);
    aligned_vector<int> send_ptr;
    aligned_vector<MPI_Request> send_requests;
    aligned_vector<int> send_indices;
    aligned_vector<char> send_buffer;
    aligned_vector<char> recv_buffer;
    aligned_vector<int> recv_rows;
    aligned_vector<int> recv_row_sizes;
    aligned_vector<int> recv_cols;
    aligned_vector<double> recv_vals;
    MPI_Status recv_status;

    int num_ints = 2*A->local_num_rows + A->local_nnz;
    int num_dbls = A->local_nnz;
    int int_bytes, dbl_bytes;
    MPI_Pack_size(num_ints, MPI_INT, MPI_COMM_WORLD, &int_bytes);
    MPI_Pack_size(num_dbls, MPI_DOUBLE, MPI_COMM_WORLD, &dbl_bytes);

    int tag = 29485;


    num_sends = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        if (proc_rows[proc] == 0)
        {
            send_procs[num_sends++] = proc;
        }
        proc_rows[proc]++;
    }
    send_procs.resize(num_sends);


    send_ptr.resize(num_sends+1);
    send_requests.resize(num_sends);
    send_ptr[0] = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        proc_to_idx[proc] = i;
        send_ptr[i+1] = send_ptr[i] + proc_rows[proc];
        proc_rows[proc] = 0;
    }

    send_indices.resize(A->local_num_rows);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        proc = partition[i];
        proc_idx = proc_to_idx[proc];
        idx = send_ptr[proc_idx] + proc_rows[proc]++;
        send_indices[idx] = i;
    }
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        proc_rows[proc] = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, proc_rows.data(), num_procs, MPI_INT, MPI_SUM,
            MPI_COMM_WORLD);
    num_recvs = proc_rows[rank];

    send_buffer.resize(int_bytes + dbl_bytes);
    ctr = 0;
    for (int i = 0; i < num_sends; i++)
    {
        prev_ctr = ctr;
        proc = partition[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        proc_rows[proc] = end - start;
        for (int j = start; j < end; j++)
        {
            row = send_indices[j];
            global_row = A->local_row_map[row];
            row_size = A->on_proc->idx1[row+1] - A->on_proc->idx1[row]
                + A->off_proc->idx1[row+1] - A->off_proc->idx1[row];

            MPI_Pack(&global_row, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                    &ctr, MPI_COMM_WORLD);
            MPI_Pack(&row_size, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                    &ctr, MPI_COMM_WORLD);

            row_start = A->on_proc->idx1[row];
            row_end = A->on_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->on_proc->idx2[k];
                global_col = A->on_proc_column_map[col];
                val = A->on_proc->vals[k];
                MPI_Pack(&global_col, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                        &ctr, MPI_COMM_WORLD);
                MPI_Pack(&val, 1, MPI_DOUBLE, send_buffer.data(), send_buffer.size(),
                        &ctr, MPI_COMM_WORLD);
            }
            row_start = A->off_proc->idx1[row];
            row_end = A->off_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                col = A->off_proc->idx2[k];
                global_col = A->off_proc_column_map[col];
                val = A->off_proc->vals[k];
                MPI_Pack(&global_col, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                        &ctr, MPI_COMM_WORLD);
                MPI_Pack(&val, 1, MPI_DOUBLE, send_buffer.data(), send_buffer.size(),
                        &ctr, MPI_COMM_WORLD);
            }
        }
        MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, MPI_PACKED, proc, tag,
                MPI_COMM_WORLD, &(send_requests[i]));
    }

    for (int i = 0; i < num_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);
        recv_buffer.resize(count);
        MPI_Recv(recv_buffer.data(), count, MPI_PACKED, proc, tag, MPI_COMM_WORLD,
                &recv_status);

        ctr = 0;

        while (ctr < count)
        {
            MPI_Unpack(recv_buffer.data(), count, &ctr, &global_row, 1, MPI_INT,
                    MPI_COMM_WORLD);
            recv_rows.push_back(global_row);
            MPI_Unpack(recv_buffer.data(), count, &ctr, &row_size, 1, MPI_INT,
                    MPI_COMM_WORLD);
            recv_row_sizes.push_back(row_size);
            for (int j = 0; j < row_size; j++)
            {
                MPI_Unpack(recv_buffer.data(), count, &ctr, &global_col, 1, MPI_INT,
                        MPI_COMM_WORLD);
                recv_cols.push_back(global_col);
                MPI_Unpack(recv_buffer.data(), count, &ctr, &val, 1, MPI_DOUBLE,
                        MPI_COMM_WORLD);
                recv_vals.push_back(val);
            }
        }
    }

    int num_rows = recv_rows.size();
    MPI_Waitall(num_sends, send_requests.data(), MPI_STATUSES_IGNORE);

    MPI_Allgather(&num_rows, 1, MPI_INT, proc_rows.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_row = 0;
    for (int i = 0; i < rank; i++)
    {
        first_row += proc_rows[i];
    }

    A_part = new ParCSRMatrix(A->global_num_rows, A->global_num_rows, num_rows, num_rows, 
            first_row, first_row, A->partition->topology);

    // Create row_ptr
    // Add values/indices to appropriate positions
    std::map<int, int> on_proc_to_local;
    for(int i = 0; i < num_rows; i++)
    {
       on_proc_to_local[recv_rows[i]] = i;
       A_part->on_proc_column_map.push_back(recv_rows[i]);
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
            col = recv_cols[ctr];
            val = recv_vals[ctr++];

            if (on_proc_to_local.find(col) != on_proc_to_local.end())
            {
                A_part->on_proc->idx2.push_back(on_proc_to_local[col]);
                A_part->on_proc->vals.push_back(val);
            }
            else
            {
                A_part->off_proc->idx2.push_back(col);
                A_part->off_proc->vals.push_back(val);
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
            A_part->off_proc_column_map.push_back(*it);
            prev_col = *it;
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


