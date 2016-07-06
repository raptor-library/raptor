// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_comm.hpp"

using namespace raptor;

/**************************************************************
*****   ParComm Initialize Sends
**************************************************************
***** Posts Isends
*****
***** Parameters
***** -------------
***** comm : MPI_Comm
*****    MPI Communicator containing all active processes
**************************************************************/
void ParComm::init_sends(const data_t* x_data, MPI_Comm comm)
{
    if (num_sends == 0) return;
    
    int send_start, send_end, send_size, proc;

    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        send_start = send_row_starts[i];
        send_end = send_row_starts[i+1];
        send_size = send_end - send_start;
        for (int j = send_start; j < send_end; j++)
        {
            send_buffer[j] = x_data[send_row_indices[j]];
        }
        MPI_Isend(&(send_buffer[send_start]), send_size, MPI_DATA_T, proc, 0, comm, &(send_requests[i]));
    }
}

/**************************************************************
*****   ParComm Initialize Recvs
**************************************************************
***** Posts Irecvs
*****
***** Parameters
***** -------------
***** comm : MPI_Comm
*****    MPI Communicator containing all active processes
**************************************************************/
void ParComm::init_recvs(MPI_Comm comm)
{
    if (num_recvs == 0) return;

    int recv_start, recv_end, recv_size, proc;

    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        recv_start = recv_col_starts[i];
        recv_end = recv_col_starts[i+1];
        recv_size = recv_end - recv_start;
        MPI_Irecv(&(recv_buffer[recv_start]), recv_size, MPI_DATA_T, proc, 0, comm, &(recv_requests[i]));
    }
}

/**************************************************************
*****   ParComm Complete Sends
**************************************************************
***** Waits for previously posted Isends to complete
**************************************************************/
void ParComm::complete_sends()
{
    if (num_sends == 0) return;
    {
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
    }
}

/**************************************************************
*****   ParComm Complete Recvs
**************************************************************
***** Waits for previously posted Irecvs to complete
**************************************************************/
void ParComm::complete_recvs()
{
    if (num_recvs == 0) return;
    {
        MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);
    }
}

/**************************************************************
*****   ParComm Initialize Sends (Transpose)
**************************************************************
***** Posts Isends (transposed, so using recv comm data)
*****
***** Parameters
***** -------------
***** comm : MPI_Comm
*****    MPI Communicator containing all active processes
**************************************************************/
void ParComm::init_sends_T(MPI_Comm comm)
{
    if (num_recvs == 0) return;
    
    int recv_start, recv_end, recv_size, proc;

    for (int i = 0; i < num_recvs; i++)
    {
        proc = recv_procs[i];
        recv_start = recv_col_starts[i];
        recv_end = recv_col_starts[i+1];
        recv_size = recv_end - recv_start;

        // Recv Buffer already should have data in it
        MPI_Isend(&(recv_buffer[recv_start]), recv_size, MPI_DATA_T, proc, 0, comm, &(recv_requests[i]));
    }
}

/**************************************************************
*****   ParComm Initialize Recvs (Transposed)
**************************************************************
***** Posts Irecvs
*****
***** Parameters
***** -------------
***** comm : MPI_Comm
*****    MPI Communicator containing all active processes
**************************************************************/
void ParComm::init_recvs_T(MPI_Comm comm)
{
    if (num_sends == 0) return;

    int send_start, send_end, send_size, proc;

    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        send_start = send_row_starts[i];
        send_end = send_row_starts[i+1];
        send_size = send_end - send_start;
        MPI_Irecv(&(send_buffer[send_start]), send_size, MPI_DATA_T, proc, 0, comm, &(send_requests[i]));
    }
}

/**************************************************************
*****   ParComm Complete Sends (Transposed)
**************************************************************
***** Waits for previously posted Isends to complete
**************************************************************/
void ParComm::complete_sends_T()
{
    if (num_recvs)
    {
        MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);
    }
}

/**************************************************************
*****   ParComm Complete Recvs (Transposed)
**************************************************************
***** Waits for previously posted Irecvs to complete
**************************************************************/
void ParComm::complete_recvs_T()
{
    if (num_sends)
    {
        MPI_Waitall(num_sends, send_requests, MPI_STATUS_IGNORE);
    }
}


/**************************************************************
*****   ParComm Init ColToProc
**************************************************************
***** Initializes col_to_proc, adding the process that corresponds
***** to each column of the off-diagonal block
*****
***** Parameters
***** -------------
***** comm_mat : MPI_Comm (const)
*****    MPI Communicator containing all active processes
***** num_cols : index_t (const)
*****    Number of columns in the off-diagonal block
***** global_to_local : std::map<index_t, index_t>&
*****    Maps global columns of the parallel matrix to local
*****    columns of the off-diagonal blocks
***** global_col_starts : Array<index_t>&
*****    Describes the partition of the parallel matrix
**************************************************************/
void ParComm::init_col_to_proc(const MPI_Comm comm_mat, const index_t num_cols, 
    std::map<index_t, index_t>& global_to_local,  
    Array<index_t>& global_col_starts)
{
    index_t proc = 0;
    index_t global_col = 0;
    index_t local_col = 0;
    col_to_proc.resize(num_cols);

    // Find the process that each col is mapped to
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

/**************************************************************
*****   ParComm Init Communicator Recvs
**************************************************************
***** Initializes the recv data for communication.
*****
***** Parameters
***** -------------
***** comm_mat : MPI_Comm (const)
*****    MPI Communicator containing all active processes
***** num_cols : index_t (const)
*****    Number of columns in the off-diagonal block
***** global_to_local : std::map<index_t, index_t>&
*****    Maps global columns of the parallel matrix to local
*****    columns of the off-diagonal blocks
**************************************************************/
void ParComm::init_comm_recvs(const MPI_Comm comm_mat, const index_t num_cols, std::map<index_t, index_t>& global_to_local)
{
    // Get MPI Information
    int rank, num_procs;
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
    ctr = 0;

    old_proc = col_to_proc[global_to_local.begin()->second];
    recv_procs.push_back(old_proc);
    recv_col_starts.push_back(ctr);

    for (std::map<index_t, index_t>::iterator i = global_to_local.begin(); i != global_to_local.end(); i++)
    {
        local_col = i->second;
        proc = col_to_proc[local_col];
        // Column lies on new processor, so add last
        // processor to communicator
        if (proc != old_proc)
        {
            recv_procs.push_back(proc);
            recv_col_starts.push_back(ctr);
            old_proc = proc;
        }
        ctr++;
    }
    // Add last processor to communicator
    recv_col_starts.push_back(ctr);

    num_recvs = recv_procs.size();
    size_recvs = recv_col_starts[num_recvs];
}

/**************************************************************
*****   ParComm Init Communicator Sends
**************************************************************
***** Initializes the send data for communication. 
*****
***** Parameters
***** -------------
***** comm_mat : MPI_Comm (const)
*****    MPI Communicator containing all active processes
***** map_to_global : Arrayn<index_t>&
*****    Maps local columns to global columns
***** global_col_starts : Array<index_t>&
*****    Describes the partition of the parallel matrix
**************************************************************/
void ParComm::init_comm_sends(const MPI_Comm comm_mat, Array<index_t>& map_to_global, Array<index_t>& global_col_starts)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    index_t recv_proc = 0;
    index_t orig_ctr = 0;
    index_t ctr = 0;
    index_t req_ctr = 0;
    index_t unsym_tag = 1212;

    // Determind number of messages I will receive
    index_t* send_buffer = NULL;
    MPI_Request* send_requests = NULL;
    MPI_Status* send_status = NULL;
    index_t* send_counts = new index_t[num_procs];
    index_t* recv_counts = new index_t[num_procs];  

    for (index_t i = 0; i < num_procs; i++)
    {
        send_counts[i] = 0;
        recv_counts[i] = 0;
    }

    if (num_recvs)
    {
        send_buffer = new index_t[size_recvs];
        send_requests = new MPI_Request[num_recvs];
        send_status = new MPI_Status[num_recvs];
        for (int i = 0; i < num_recvs; i++)
        {
            send_requests[i] = MPI_REQUEST_NULL;
        }

        //Send everything in recv_idx[recv_proc] to recv_proc;
        int recv_start, recv_end;
        for (index_t i = 0; i < num_recvs; i++)
        {
            orig_ctr = ctr;
            recv_proc = recv_procs[i];
            recv_start = recv_col_starts[i];
            recv_end = recv_col_starts[i+1];
            for (int j = recv_start; j < recv_end; j++)
            {
                send_buffer[ctr++] = map_to_global[j];
            }
            MPI_Isend(&send_buffer[orig_ctr], ctr - orig_ctr, MPI_INDEX_T, recv_proc, unsym_tag, comm_mat, &send_requests[i]);
            send_counts[recv_proc] = 1;
        }

    }

    // AllReduce - sum number of sends to each process
    MPI_Allreduce(send_counts, recv_counts, num_procs, MPI_INDEX_T, MPI_SUM, comm_mat);
    delete[] send_counts;
    delete[] recv_counts;

    int count = 0;
    int avail_flag;
    MPI_Status recv_status;
    send_row_starts.push_back(0);
    while (send_procs.size() < num_recvs)
    {
        //Probe for messages, and recv any found
        MPI_Iprobe(MPI_ANY_SOURCE, unsym_tag, comm_mat, &avail_flag, &recv_status);
        if (avail_flag)
        {
            MPI_Get_count(&recv_status, MPI_INDEX_T, &count);
            index_t recv_buffer[count];
            MPI_Recv(&recv_buffer, count, MPI_INDEX_T, MPI_ANY_SOURCE, unsym_tag, comm_mat, &recv_status);
            send_procs.push_back(recv_status.MPI_SOURCE);
            for (int i = 0; i < count; i++)
            {
                send_row_indices.push_back(recv_buffer[i] - global_col_starts[rank]);
            }
            send_row_starts.push_back(send_row_indices.size());
        }
    }

    if (num_recvs)
    {
        MPI_Waitall(num_recvs, send_requests, send_status);

        delete[] send_buffer;
        delete[] send_requests;
        delete[] send_status;
    }

    num_sends = send_procs.size();
    size_sends = send_row_starts[num_sends];

}
