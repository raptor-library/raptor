// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "repartition.hpp"

void make_contiguous(ParCSRMatrix* A, aligned_vector<int>& off_proc_part_map)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    std::map<int, int> global_to_local;
    aligned_vector<int> proc_num_cols(num_procs);
    aligned_vector<int> recvvec;

    int ctr = 0;
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

    A->comm = new ParComm(A->partition->topology, A->off_proc_column_map,
            off_proc_part_map, A->local_row_map);

    for (int i = 0; i < A->local_num_rows; i++)
    {
        A->on_proc_column_map[i] = A->partition->first_local_col + i;
    }
    A->local_row_map = A->get_on_proc_column_map();
    recvvec = A->comm->communicate(A->local_row_map);
    for (int i = 0; i < A->off_proc_num_cols; i++)
        A->off_proc_column_map[i] = recvvec[i];


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
    int recv_size;
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
    aligned_vector<int> recv_row_ptr;
    aligned_vector<int> recv_cols;
    aligned_vector<double> recv_vals;
    MPI_Status recv_status;

    int num_ints = 2*A->local_num_rows + A->local_nnz;
    int num_dbls = A->local_nnz;
    int int_bytes, dbl_bytes;
    MPI_Pack_size(num_ints, MPI_INT, MPI_COMM_WORLD, &int_bytes);
    MPI_Pack_size(num_dbls, MPI_DOUBLE, MPI_COMM_WORLD, &dbl_bytes);

    int tag = 29485;

    aligned_vector<int> off_parts(A->off_proc_num_cols);
    aligned_vector<int>& recvvec = A->comm->communicate(partition);
    std::copy(recvvec.begin(), recvvec.end(), off_parts.begin());

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

    // TODO -- send partitions for each global col (both on and off proc) if part[row] != part[col]
    aligned_vector<int> col_bool(A->local_num_rows, 0);
    aligned_vector<int> off_col_bool(A->off_proc_num_cols, 0);
    aligned_vector<int> send_cols(A->local_num_rows);
    aligned_vector<int> off_send_cols(A->off_proc_num_cols);
    int n_cols, off_n_cols;
    int n_rows, part;
    int off_col_size = 2 * (A->off_proc_num_cols + A->local_num_rows) * num_sends;
    int off_col_bytes, row_bytes;
    MPI_Pack_size(off_col_size, MPI_INT, MPI_COMM_WORLD, &off_col_bytes);
    MPI_Pack_size(num_sends, MPI_INT, MPI_COMM_WORLD, &row_bytes);
    send_buffer.resize(int_bytes + dbl_bytes + off_col_bytes + row_bytes);
    ctr = 0;
    for (int i = 0; i < num_sends; i++)
    {
        prev_ctr = ctr;
        proc = partition[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        n_rows = end - start;
        proc_rows[proc] = n_rows;
        MPI_Pack(&n_rows, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                &ctr, MPI_COMM_WORLD);
        n_cols = 0;
        off_n_cols = 0;
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
                if (partition[col] != proc && col_bool[col] == 0)
                {
                    send_cols[n_cols++] = col;
                    col_bool[col] = 1;
                }
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
                if (off_parts[col] != proc && off_col_bool[col] == 0)
                {
                    off_send_cols[off_n_cols++] = col;
                    off_col_bool[col] = 1;
                }
            }
        }
        for (int j = 0; j < n_cols; j++)
        {
            col = send_cols[j];
            col_bool[col] = 0;
            global_col = A->local_row_map[col];
            part = partition[col];
            MPI_Pack(&global_col, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                    &ctr, MPI_COMM_WORLD);
            MPI_Pack(&part, 1, MPI_INT, send_buffer.data(), send_buffer.size(), &ctr,
                    MPI_COMM_WORLD);
        }
        for (int j = 0; j < off_n_cols; j++)
        {
            col = off_send_cols[j];
            off_col_bool[col] = 0;
            global_col = A->off_proc_column_map[col];
            part = off_parts[col];
            MPI_Pack(&global_col, 1, MPI_INT, send_buffer.data(), send_buffer.size(),
                    &ctr, MPI_COMM_WORLD);
            MPI_Pack(&part, 1, MPI_INT, send_buffer.data(), send_buffer.size(), &ctr,
                    MPI_COMM_WORLD);
        }
        MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, MPI_PACKED, proc, tag,
                MPI_COMM_WORLD, &(send_requests[i]));
    }

    std::map<int,int> off_proc_to_local;
    aligned_vector<int> off_col_to_global;
    aligned_vector<int> off_col_parts;
    recv_size = 0;
    recv_row_ptr.push_back(recv_size);
    for (int i = 0; i < num_recvs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_PACKED, &count);
        recv_buffer.resize(count);
        MPI_Recv(recv_buffer.data(), count, MPI_PACKED, proc, tag, MPI_COMM_WORLD,
                &recv_status);

        ctr = 0;

        MPI_Unpack(recv_buffer.data(), count, &ctr, &n_rows, 1, MPI_INT,
                MPI_COMM_WORLD);
        for (int j = 0; j < n_rows; j++)
        {
            MPI_Unpack(recv_buffer.data(), count, &ctr, &global_row, 1, MPI_INT,
                    MPI_COMM_WORLD);
            recv_rows.push_back(global_row);
            MPI_Unpack(recv_buffer.data(), count, &ctr, &row_size, 1, MPI_INT,
                    MPI_COMM_WORLD);
            recv_size += row_size;
            recv_row_ptr.push_back(recv_size);
            for (int k = 0; k < row_size; k++)
            {
                MPI_Unpack(recv_buffer.data(), count, &ctr, &global_col, 1, MPI_INT,
                        MPI_COMM_WORLD);
                recv_cols.push_back(global_col);
                MPI_Unpack(recv_buffer.data(), count, &ctr, &val, 1, MPI_DOUBLE,
                        MPI_COMM_WORLD);
                recv_vals.push_back(val);
            }
        }
        while (ctr < count)
        {
            MPI_Unpack(recv_buffer.data(), count, &ctr, &global_col, 1, MPI_INT,
                    MPI_COMM_WORLD);
            MPI_Unpack(recv_buffer.data(), count, &ctr, &part, 1, MPI_INT, MPI_COMM_WORLD);
            if (off_proc_to_local.find(global_col) == off_proc_to_local.end())
            {
                off_proc_to_local[global_col] = off_col_to_global.size();
                off_col_to_global.push_back(global_col);
                off_col_parts.push_back(part);
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
    A_part->off_proc_num_cols = off_col_parts.size();
    A_part->off_proc_column_map.resize(A_part->off_proc_num_cols);
    A_part->on_proc_column_map.resize(A_part->local_num_rows);

    aligned_vector<int> off_proc_part_map(A_part->off_proc_num_cols);
    aligned_vector<int> off_col_order(A_part->off_proc_num_cols);
    std::iota(off_col_order.begin(), off_col_order.end(), 0);
    std::sort(off_col_order.begin(), off_col_order.end(), 
            [&](const int i, const int j)
            {
                if (off_col_parts[i] == off_col_parts[j])
                    return off_col_to_global[i] < off_col_to_global[j];
                return off_col_parts[i] < off_col_parts[j];
            });
    for (int i = 0; i < A_part->off_proc_num_cols; i++)
    {
        col = off_col_order[i];
        global_col = off_col_to_global[col];
        off_proc_to_local[global_col] = i;
        A_part->off_proc_column_map[i] = global_col;
        off_proc_part_map[i] = off_col_parts[col];
    }

    // Create row_ptr
    // Add values/indices to appropriate positions
    aligned_vector<int> row_order(A_part->local_num_rows);
    std::iota(row_order.begin(), row_order.end(), 0);
    std::sort(row_order.begin(), row_order.end(),
            [&](const int i, const int j)
            {
                return recv_rows[i] < recv_rows[j];
            });
    std::map<int, int> on_proc_to_local;
    for(int i = 0; i < num_rows; i++)
    {
        row = row_order[i];
        global_row = recv_rows[row];
        on_proc_to_local[global_row] = i;
        A_part->on_proc_column_map[i] = global_row;
    }
    A_part->local_row_map = A_part->get_on_proc_column_map();
    A_part->on_proc_num_cols = A_part->on_proc_column_map.size();

    A_part->on_proc->idx1[0] = 0;
    A_part->off_proc->idx1[0] = 0;
    for (int i = 0; i < num_rows; i++)
    {
        row = row_order[i];
        row_start = recv_row_ptr[row];
        row_end = recv_row_ptr[row+1];
        for (int j = row_start; j < row_end; j++)
        {
            col = recv_cols[j];
            val = recv_vals[j];

            if (on_proc_to_local.find(col) != on_proc_to_local.end())
            {
                A_part->on_proc->idx2.push_back(on_proc_to_local[col]);
                A_part->on_proc->vals.push_back(val);
            }
            else
            {
                A_part->off_proc->idx2.push_back(off_proc_to_local[col]);
                A_part->off_proc->vals.push_back(val);
            }
        }
        A_part->on_proc->idx1[i+1] = A_part->on_proc->idx2.size();
        A_part->off_proc->idx1[i+1] = A_part->off_proc->idx2.size();
    }
    A_part->on_proc->nnz = A_part->on_proc->idx2.size();
    A_part->off_proc->nnz = A_part->off_proc->idx2.size();
    A_part->local_nnz = A_part->on_proc->nnz + A_part->off_proc->nnz;

    new_local_rows.resize(A_part->on_proc_num_cols);
    std::copy(A_part->on_proc_column_map.begin(), A_part->on_proc_column_map.end(),
            new_local_rows.begin());

    make_contiguous(A_part, off_proc_part_map);

    return A_part;
}


