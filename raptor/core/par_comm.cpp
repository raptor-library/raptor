// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_comm.hpp"

void ParComm::init_col_to_proc(MPI_Comm comm_mat, index_t num_cols, 
    std::map<index_t, index_t> global_to_local,  
    index_t* global_col_starts)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    index_t proc = 0;
    index_t global_col = 0;
    index_t local_col = 0;
    col_to_proc.reserve(num_cols);
    for (std::map<index_t, index_t>::iterator i = global_to_local.begin(); i != global_to_local.end(); i++)
    {
        global_col = i->first;
        local_col = i->second;
        while (global_col >= global_col_starts[proc+1])
        {
            proc++;
        }
        col_to_proc[local_col] = proc;
    }
}

void ParComm::init_comm_recvs(MPI_Comm comm_mat, index_t num_cols,   
    std::map<index_t, index_t> global_to_local)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    std::vector<index_t>    proc_cols;
    index_t proc;
    index_t old_proc;
    index_t first;
    index_t last;
    index_t local_col;
    index_t ctr;

    //Find inital proc (local col 0 lies on)
    proc = col_to_proc[0];

    // For each offd col, find proc it lies on.  Add proc and list
    // of columns it holds to map recvIndices
    last = 0;
    old_proc = col_to_proc[global_to_local.begin()->second];

    ctr = 0;
    for (std::map<index_t, index_t>::iterator i = global_to_local.begin(); i != global_to_local.end(); i++)
    {
        local_col = i->second;
        proc_cols.push_back(local_col);
        proc = col_to_proc[local_col];
        // Column lies on new processor, so add last
        // processor to communicator
        if (proc != old_proc)
        {
            first = last;
            last = ctr;
            std::vector<index_t> newvec(proc_cols.begin() + first, proc_cols.begin() + last);
            recv_indices[old_proc] = newvec;
            recv_procs.push_back(old_proc);
        }
        old_proc = proc;
        ctr++;
    }
    // Add last processor to communicator
    first = last;
    std::vector<index_t> newvec(proc_cols.begin() + first, proc_cols.begin() + num_cols);
    recv_indices[old_proc] = newvec;
    recv_procs.push_back(old_proc);

    //Store total number of values to be sent/received
    size_recvs = num_cols;
}

void ParComm::init_comm_sends_sym_csr(MPI_Comm comm_mat, Matrix* offd, std::map<index_t, index_t> global_to_local)
{
    index_t* ptr;
    index_t* idx;
    index_t num_rows;
    index_t row_start;
    index_t row_end;
    index_t proc;
    index_t old_proc;
    index_t local_col;

    // Get CSR Matrix variables
    ptr = offd->indptr.data();
    idx = offd->indices.data();
    num_rows = offd->n_rows;
    size_sends = 0;
    for (index_t i = 0; i < num_rows; i++)
    {
        row_start = ptr[i];
        row_end = ptr[i+1];
        if (row_start == row_end) 
        {
            continue;
        }
        old_proc = col_to_proc[idx[row_start]];
        for (index_t j = row_start; j < row_end; j++)
        {
            local_col = idx[j];
            proc = col_to_proc[local_col];
            // Column lies on new processor, so add last
            // processor to communicator
            if (proc != old_proc)
            {
                if (send_indices.count(old_proc))
                {
                    if (send_indices[old_proc].back() != i)
                    {
                        send_indices[old_proc].push_back(i);   
                    }
                }
                else
                {
                    std::vector<index_t> tmp;
                    tmp.push_back(i);
                    send_indices[old_proc] = tmp;
                    send_procs.push_back(old_proc);
                }
                size_sends++;
            }
            old_proc = proc;
        }
        // Add last processor to communicator
        if (send_indices.count(old_proc))
        {
            if (send_indices[old_proc].back() != i)
            {
                send_indices[old_proc].push_back(i);
            }
        }
        else
        {
            std::vector<index_t> tmp;
            tmp.push_back(i);
            send_indices[old_proc] = tmp;
            send_procs.push_back(old_proc);
        }
        size_sends++;
    }
}

void ParComm::init_comm_sends_sym_csc(MPI_Comm comm_mat, Matrix* offd, std::map<index_t, index_t> global_to_local)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    index_t* ptr;
    index_t* idx;
    index_t num_cols;
    index_t col_start;
    index_t col_end;
    index_t proc;
    index_t old_proc;
    index_t local_col;
    std::vector<index_t>::iterator it;

    // Get CSR Matrix variables
    ptr = offd->indptr.data();
    idx = offd->indices.data();
    num_cols = offd->n_cols;
    size_sends = 0;
    
    old_proc = col_to_proc[global_to_local.begin()->second];
    for (std::map<index_t, index_t>::iterator i = global_to_local.begin(); i != global_to_local.end(); i++)
    {
        local_col = i->second;
        col_start = ptr[local_col];
        col_end = ptr[local_col+1];
        if (col_start == col_end) 
        {
            continue;
        }
            
        proc = col_to_proc[local_col];

        if (proc != old_proc)
        {
            send_procs.push_back(old_proc);
            std::sort(send_indices[old_proc].begin(), send_indices[old_proc].end());
            it = std::unique(send_indices[old_proc].begin(), send_indices[old_proc].end());
            send_indices[old_proc].resize(std::distance(send_indices[old_proc].begin(), it));
            size_sends += send_indices[old_proc].size();

            std::vector<index_t> tmp;
            send_indices[proc] = tmp;
        }

        for (index_t j = col_start; j < col_end; j++)
        {
            send_indices[proc].push_back(idx[j]);
        }
        old_proc = proc;
    }
    send_procs.push_back(old_proc);
    std::sort(send_indices[old_proc].begin(), send_indices[old_proc].end());
    it = std::unique(send_indices[old_proc].begin(), send_indices[old_proc].end());
    send_indices[old_proc].resize(std::distance(send_indices[old_proc].begin(), it));
    size_sends += send_indices[old_proc].size();
}

void ParComm::init_comm_sends_unsym(MPI_Comm comm_mat, std::vector<index_t> map_to_global, index_t* global_col_starts)
{
    // Get MPI Information
    index_t rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    index_t recv_proc = 0;
    index_t orig_ctr = 0;
    index_t ctr = 0;
    index_t req_ctr = 0;
    index_t unsym_tag = 1212;
    index_t recv_size = recv_procs.size();
    size_sends = 0;

    index_t* send_buffer = new index_t[size_recvs];
    MPI_Request* send_requests = new MPI_Request[recv_size];
    MPI_Status* send_status = new MPI_Status[recv_size];
    for (int i = 0; i < recv_size; i++)
    {
        send_requests[i] = MPI_REQUEST_NULL;
    }

    index_t num_recvs;

    // Determind number of messages I will receive
    index_t* send_counts = new index_t[num_procs];
    index_t* recv_counts = new index_t[num_procs];  

    for (index_t i = 0; i < num_procs; i++)
    {
        send_counts[i] = 0;
        recv_counts[i] = 0;
    }

    //Send everything in recv_idx[recv_proc] to recv_proc;
    for (index_t i = 0; i < recv_size; i++)
    {
        orig_ctr = ctr;
        recv_proc = recv_procs[i];
        for (auto recv_idx : recv_indices[recv_proc])
        {
            send_buffer[ctr++] = map_to_global[recv_idx];
        }
        MPI_Isend(&send_buffer[orig_ctr], ctr - orig_ctr, MPI_INT, recv_proc, unsym_tag, comm_mat, &send_requests[i]);
        send_counts[recv_proc] = 1;
    }

    // AllReduce - sum number of sends to each process
    MPI_Allreduce(send_counts, recv_counts, num_procs, MPI_INT, MPI_SUM, comm_mat);
    num_recvs = recv_counts[rank];
    delete[] send_counts;
    delete[] recv_counts;

    index_t count = 0;
    index_t avail_flag;
    MPI_Status recv_status;
    while (send_procs.size() < num_recvs)
    {
        //Probe for messages, and recv any found
        MPI_Iprobe(MPI_ANY_SOURCE, unsym_tag, comm_mat, &avail_flag, &recv_status);
        if (avail_flag)
        {
            MPI_Get_count(&recv_status, MPI_INT, &count);
            index_t recv_buffer[count];
            MPI_Recv(&recv_buffer, count, MPI_INT, MPI_ANY_SOURCE, unsym_tag, comm_mat, &recv_status);
            for (int i = 0; i < count; i++) recv_buffer[i] = recv_buffer[i] - global_col_starts[rank];
            send_procs.push_back(recv_status.MPI_SOURCE);
            std::vector<index_t> send_idx(recv_buffer, recv_buffer + count);
            send_indices[recv_status.MPI_SOURCE] = send_idx;
            size_sends += send_idx.size();
        }
    }

    MPI_Waitall(recv_procs.size(), send_requests, send_status);
   
    delete[] send_buffer;
    delete[] send_requests;
    delete[] send_status;

}
