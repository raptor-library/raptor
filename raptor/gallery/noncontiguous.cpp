// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "noncontiguous.hpp"
#include <string>
#include <set>

ParMatrix* non_contiguous(int global_rows, int global_cols, 
        std::vector<coo_data>& matrix_data)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    ParMatrix* A;
    std::map<int, int> row_global_to_local;
    std::vector<int> row_local_to_global;
    std::vector<int> col_local_to_global;
    std::vector<std::pair<int, int>> proc_row_order;
    std::set<int> row_global_set;
    std::set<int> col_global_set;
    std::vector<int> local_to_global_tmp;
    int* proc_row_sizes;
    int* global_row_data;

    int ctr;
    int lcl_nnz = matrix_data.size();
    int proc_start, proc_end;
    int global_row, global_col;
    int row, col;

    int max_local_rows;
    int* buffer;

    A = new ParMatrix();
    A->global_rows = global_rows;
    A->global_cols = global_cols;
    A->comm_mat = MPI_COMM_WORLD;

    for (int i = 0; i < lcl_nnz; i++)
    {
        // TODO -- does this need to be passed by reference?
        coo_data& tmp = matrix_data[i];
        row_global_set.insert(tmp.row);
        col_global_set.insert(tmp.col);
    }

    ctr = 0;
 //   row_local_to_global.reserve(row_global_set.size());
 //   col_local_to_global.reserve(col_global_set.size());
    for (std::set<int>::iterator it = row_global_set.begin();
            it != row_global_set.end(); ++it)
    {
        row_local_to_global.push_back(*it);
        row_global_to_local[*it] = ctr++;
    }
    for (std::set<int>::iterator it = col_global_set.begin();
            it != col_global_set.end(); ++it)
    {
        if (row_global_to_local.find(*it) == row_global_to_local.end())
        {
            col_local_to_global.push_back(*it);
        }
    }

    A->local_rows = row_global_set.size();
    A->local_cols = A->local_rows;
    A->offd_num_cols = col_local_to_global.size();

    A->diag = new Matrix(A->local_rows, A->local_cols, CSR);
    A->offd = new Matrix(A->local_rows, A->offd_num_cols, CSC);

    proc_row_sizes = new int[num_procs];
    MPI_Allgather(&(A->local_rows), 1, MPI_INT, proc_row_sizes, 1,
           MPI_INT, MPI_COMM_WORLD);
    A->global_col_starts.resize(num_procs+1);
    A->global_col_starts[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        A->global_col_starts[i+1] = A->global_col_starts[i] + proc_row_sizes[i];
    }
    A->first_row = A->global_col_starts[rank];
    A->first_col_diag = A->first_row;
    
    global_row_data = new int[A->global_rows];
    MPI_Allgatherv(row_local_to_global.data(), A->local_rows, MPI_INT, global_row_data, 
            proc_row_sizes, A->global_col_starts.data(), MPI_INT, MPI_COMM_WORLD);

    // TODO - instead of MPI_Allgatherv, need to send/recv only with neighbor 
    // so that only a small portion of data is on process at once
    max_local_rows = 0;
    for (int i = 0; i < num_procs; i++)
    {
        int lcl_rows = A->global_col_starts[i+1] - A->global_col_starts[i];
        if (lcl_rows > max_local_rows) max_local_rows = lcl_rows;
    }
    buffer = new int[2*max_local_rows];

    // Add my rows to buffer
    ctr = 0;
    for (std::set<int>::iterator it = row_global_set.begin();
            it != row_global_set.end(); ++it)
    {
        buffer[ctr++] = *it;
    }

    int* send_buffer;
    int* recv_buffer;
    int* tmp_buffer;
    int send_proc, recv_proc;
    int current_proc, next_proc;
    int send_size, recv_size;
    
    int tag = 9876;

    send_buffer = &(buffer[0]);
    recv_buffer = &(buffer[max_local_rows]);
    send_proc = (rank + 1) % num_procs;
    recv_proc = (rank - 1);
    if (recv_proc < 0) recv_proc = num_procs - 1;
    current_proc = rank;
    next_proc = recv_proc;

    for (int i = 0; i < num_procs - 1; i++)
    {
        send_size = A->global_col_starts[current_proc+1] 
            - A->global_col_starts[current_proc];

        recv_size = A->global_col_starts[next_proc+1] 
            - A->global_col_starts[next_proc];

        if (rank % 2 == 0)
        {
            MPI_Send(send_buffer, send_size, MPI_INT, send_proc, tag, 
                    MPI_COMM_WORLD);
            MPI_Recv(recv_buffer, recv_size, MPI_INT, recv_proc, tag, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            MPI_Recv(recv_buffer, recv_size, MPI_INT, recv_proc, tag, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buffer, send_size, MPI_INT, send_proc, tag, 
                    MPI_COMM_WORLD);
        }

        // Recv buffer contains rows that are stored on current proc

        if (A->offd_num_cols)
        {
            proc_start = A->global_col_starts[current_proc];
            proc_row_order.resize(recv_size);
            for (int j = 0; j < recv_size; j++)
            {
                proc_row_order[j] = std::make_pair(recv_buffer[j], j + proc_start);
            }

            std::sort(proc_row_order.begin(), proc_row_order.end(),
                    [](const std::pair<int, int>& lhs, 
                        const std::pair<int, int>& rhs)
                    { return lhs.first < rhs.first; } );

            ctr = 0;
            for (std::vector<std::pair<int, int>>::iterator it = proc_row_order.begin();
                    it != proc_row_order.end(); ++it)
            {
                while (ctr + 1 < A->offd_num_cols &&
                        col_local_to_global[ctr + 1] <= it->first)
                {
                    ctr++;
                }
                if (col_local_to_global[ctr] == it->first)
                {
                    A->global_to_local[it->second] = ctr;
                    local_to_global_tmp.push_back(it->second);
                }
            }
        }

        tmp_buffer = &(send_buffer[0]);
        send_buffer = recv_buffer;
        recv_buffer = tmp_buffer;

        current_proc = next_proc;
        next_proc = next_proc - 1;
        if (next_proc < 0) next_proc = num_procs - 1;

    }

    std::sort(local_to_global_tmp.begin(), local_to_global_tmp.end());
    for (int j = 0; j < A->offd_num_cols; j++)
    {
        global_col = local_to_global_tmp[j];
        std::map<int, int>::iterator map_it = A->global_to_local.find(global_col);
        map_it->second = j;
        A->local_to_global.push_back(global_col);
    }

    for (int i = 0; i < lcl_nnz; i++)
    {
        // TODO -- does this need to be passed by reference?
        coo_data& tmp = matrix_data[i];
        row = row_global_to_local[tmp.row];
        if (row_global_to_local.find(tmp.col) != row_global_to_local.end())
        {
            col = row_global_to_local[tmp.col];
            A->diag->add_value(row, col, tmp.value);
        }
        else
        {
            col = A->global_to_local[col];
            A->offd->add_value(row, col, tmp.value);
        }
    }

    A->diag->finalize();
    if (A->offd_num_cols) A->offd->finalize();
    else delete A->offd;

    if (A->local_rows)
    {
        A->comm = new ParComm(A->local_to_global, A->global_col_starts, A->first_row);
    }
    else
    {
        A->comm = new ParComm();
        A->comm->num_sends = 0;
        A->comm->num_recvs = 0;
    }
    A->local_nnz = A->diag->nnz;
    if (A->offd_num_cols) A->local_nnz += A->offd->nnz;

    delete[] buffer;
    delete[] global_row_data;
    delete[] proc_row_sizes;

    return A;
}
