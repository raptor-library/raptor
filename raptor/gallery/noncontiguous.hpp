// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef NON_CONTIGUOUS_HPP
#define NON_CONTIGUOUS_HPP

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

// Acoo holds local values... nothing actually needs to be changed here
// Only convert on_proc_column_map to first_local_row + local_num_rows 
// Find off_proc_column_map from values from assumed proc associatedd
// with each off proc column

//ParCSRMatrix* noncontiguous(std::vector<int>& row_indices, std::vector<int>& col_indices,
//        std::vector<double>& values, std::vector<int>& on_proc_column_map,
//        std::vector<int>& off_proc_column_map)
ParCSRMatrix* noncontiguous(ParCOOMatrix* A_coo, std::vector<int>& on_proc_column_map)
{
    int rank;
    int num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_nnz = A_coo->local_nnz;
    int assumed_num_cols, assumed_first_col, assumed_last_col;
    int local_assumed_num_cols;
    int assumed_proc, orig_col, new_col;
    int proc, proc_idx, idx, ctr;
    int num_sends, n_sent, send_size;
    int size_recvs;
    int send_key, recv_key;
    int row, col, cur_row;
    int start, end;
    int assumed_col, local_col;
    int msg_avail, finished, count;
    MPI_Request barrier_request;
    double val;

    std::vector<int> assumed_col_to_new;

    std::vector<int> proc_num_cols(num_procs);
    std::vector<int> send_procs;
    std::vector<int> send_ptr;
    std::vector<int> send_ctr;
    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;
    std::vector<MPI_Request> send_requests;

    std::vector<MPI_Request> recv_requests;

    std::map<int, int> global_to_local;
    ctr = 0;
    for (std::vector<int>::iterator it = A_coo->off_proc_column_map.begin();  
            it != A_coo->off_proc_column_map.end(); ++it)
    {
        global_to_local[*it] = ctr++;
    }

    // Find how many columns are local to each process
    MPI_Allgather(&(A_coo->on_proc_num_cols), 1, MPI_INT, proc_num_cols.data(), 1, MPI_INT,
            MPI_COMM_WORLD);

    // Determine the new first local row / first local col of rank
    A_coo->partition->first_local_col = 0;
    for (int i = 0; i < rank; i++)
    {
        A_coo->partition->first_local_col += proc_num_cols[i];
    }
    A_coo->partition->first_local_row = A_coo->partition->first_local_col;

    // Determine the global number of columns and rows
    A_coo->global_num_cols = A_coo->partition->first_local_col;
    for (int i = rank; i < num_procs; i++)
    {
        A_coo->global_num_cols += proc_num_cols[i];
    }
    A_coo->global_num_rows = A_coo->global_num_cols;

    // Determine which columns are assumed local to rank
    assumed_num_cols = ((A_coo->global_num_cols - 1) / num_procs) + 1;
    assumed_first_col = assumed_num_cols * rank;
    assumed_last_col = assumed_first_col + assumed_num_cols;
    if (assumed_first_col > A_coo->global_num_cols)
        assumed_first_col = A_coo->global_num_cols;
    if (assumed_last_col > A_coo->global_num_cols)
        assumed_last_col = A_coo->global_num_cols;
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
    for (int i = 0; i < A_coo->on_proc_num_cols; i++)
    {
        orig_col = on_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        proc_num_cols[assumed_proc]++;
    }

    // Re-arrange on_proc cols, ordered by assumed proc
    send_size = 0;
    send_ptr.push_back(send_size);
    for (int i = 0; i < num_procs; i++)
    {
        if (proc_num_cols[i])
        {
            send_size += proc_num_cols[i];
            proc_num_cols[i] = send_procs.size();
            send_procs.push_back(i);
            send_ptr.push_back(send_size);
        }
    }
    num_sends = send_procs.size();
    if (send_size)
    {
        send_ctr.resize(num_sends, 0);
        send_buffer.resize(2*send_size);
        send_requests.resize(num_sends);
        for (int i = 0; i < A_coo->on_proc_num_cols; i++)
        {
            orig_col = on_proc_column_map[i];
            new_col = A_coo->partition->first_local_col + i;
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
    for (int i = 0; i < A_coo->off_proc_num_cols; i++)
    {
        orig_col = A_coo->off_proc_column_map[i];
        assumed_proc = orig_col / assumed_num_cols;
        proc_num_cols[assumed_proc]++;
    }

    // Create send_procs, send_ptr
    send_size = 0;
    send_ptr.push_back(send_size);
    for (int i = 0; i < num_procs; i++)
    {
        if (proc_num_cols[i])
        {
            send_size += proc_num_cols[i];
            proc_num_cols[i] = send_procs.size();
            send_procs.push_back(i);
            send_ptr.push_back(send_size);
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
    for (int i = 0; i < A_coo->off_proc_num_cols; i++)
    {
        orig_col = A_coo->off_proc_column_map[i];
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
                A_coo->off_proc_column_map[local_col] = new_col;
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
                A_coo->off_proc_column_map[local_col] = new_col;
            }
        }
    }

    // re-index columns of off_proc (ordered by new global columns)
    if (A_coo->off_proc_num_cols)
    {
        // Find permutation of off_proc columns, sorted by global 
        // column indices in ascending order
        std::vector<int> p(A_coo->off_proc_num_cols);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), 
                [&](int i, int j)
                {
                    return A_coo->off_proc_column_map[i] < A_coo->off_proc_column_map[j];
                });
    
        // Form off_proc_orig_to_new, mapping original off_proc local
        // column indices to new local column indices
        std::vector<int> off_proc_orig_to_new(A_coo->off_proc_num_cols);
        for (int i = 0; i < A_coo->off_proc_num_cols; i++)
        {
            off_proc_orig_to_new[p[i]] = i;
        }

        // Re-index columns of off_proc
        for (std::vector<int>::iterator it = A_coo->off_proc->idx2.begin();
                it != A_coo->off_proc->idx2.end(); ++it)
        {
            *it = off_proc_orig_to_new[*it];
        }

        // Sort off_proc_column_map based on permutation vector p
        std::vector<bool> done(A_coo->off_proc_num_cols);
        for (int i = 0; i < A_coo->off_proc_num_cols; i++)
        {
            if (done[i]) continue;

            done[i] = true;
            int prev_j = i;
            int j = p[i];
            while (i != j)
            {
                std::swap(A_coo->off_proc_column_map[prev_j], A_coo->off_proc_column_map[j]);
                done[j] = true;
                prev_j = j;
                j = p[j];
            }
        }
    }

    A_coo->comm = new ParComm(A_coo->partition, A_coo->off_proc_column_map);
    ParCSRMatrix* A = new ParCSRMatrix(A_coo);

    // Sort rows, removing duplicate entries and moving diagonal 
    // value to first
    if (A->on_proc->nnz)
    {
        A->on_proc->sort();
    }

    if (A->off_proc->nnz)
    {
        A->off_proc->sort();
    }

    return A;
}
           

#endif
