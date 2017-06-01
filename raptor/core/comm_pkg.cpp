// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_pkg.hpp"

using namespace raptor;

int CommPkg::form_send_data(const int part,
        const std::vector<int>& column_map,
        std::vector<int>& send_procs,
        std::vector<int>& send_ptr,
        std::vector<int>& send_indices)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int proc, ctr;
    int size, idx;

    send_indices.resize(column_map.size());

    // Find what number of my local columns are assumed to be
    // local columns to each proc
    std::vector<int> proc_sizes(num_procs, 0);
    for (std::vector<int>::const_iterator it = column_map.begin();
            it != column_map.end(); ++it)
    {
        proc = *it / part;
        proc_sizes[proc]++;
    }

    // Create send info
    ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        size = proc_sizes[i];
        if (size)
        {
            send_procs.push_back(i);
            send_ptr.push_back(ctr);
            proc_sizes[i] = ctr;
            ctr += size;
        }
    }   
    send_ptr.push_back(ctr);

    // Add global cols to send indices in appropriate locations
    for (std::vector<int>::const_iterator it = column_map.begin();
            it != column_map.end(); ++it)
    {
        proc = *it / part;
        idx = proc_sizes[proc]++;
        send_indices[idx] = *it;
    }

    return send_procs.size();
}


void CommPkg::form_col_to_proc(const int global_num_cols,
        const std::vector<int>& on_proc_column_map,
        const std::vector<int>& off_proc_column_map,
        std::vector<int>& off_proc_col_to_proc)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int start, end;
    int local_col, global_col;
    int idx, proc, ctr, prev_ctr;
    int n_recvs, count;
    std::vector<int> assumed_col_procs;
    int assumed_part = find_proc_col_starts(global_num_cols, 
            on_proc_column_map, assumed_col_procs);
    int assumed_first = assumed_part * rank;
    int assumed_num_cols = assumed_col_procs.size();
    int off_proc_num_cols = off_proc_column_map.size();
    std::map<int, int> global_to_local;
    std::vector<int> send_procs;
    std::vector<int> send_ptr;
    std::vector<int> send_indices;
    std::vector<MPI_Request> send_requests;
    MPI_Status recv_status;

    int num_sends = form_send_data(assumed_part, off_proc_column_map, send_procs, 
            send_ptr, send_indices);
    if (num_sends)
    {
        send_requests.resize(num_sends);
    }
    if (off_proc_num_cols)
    {
        off_proc_col_to_proc.resize(off_proc_num_cols);
    }

    ctr = 0;
    for (std::vector<int>::const_iterator it = off_proc_column_map.begin(); 
            it != off_proc_column_map.end(); ++it)
    {
        global_to_local[*it] = ctr++;
    }

    int n_sends = 0;
    int size_recvd = 0;
    std::vector<int> recv_indices(send_indices.size());
    std::vector<MPI_Request> recv_requests(send_requests.size());
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        if (proc != rank)
        {
            MPI_Issend(&(send_indices[start]), end - start, MPI_INT, proc, 2468, 
                    MPI_COMM_WORLD, &(send_requests[n_sends]));
            MPI_Irecv(&(recv_indices[start]), end - start, MPI_INT, proc, 3579, 
                    MPI_COMM_WORLD, &(recv_requests[n_sends++]));

        }
        else
        {
            size_recvd += (end - start);
        }
    }

    int finished, msg_avail;
    MPI_Request barrier_request;
    if (n_sends)
    {
        MPI_Testall(n_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        while(!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, 2468, MPI_COMM_WORLD, &msg_avail, &recv_status);
            if (msg_avail)
            {
                MPI_Get_count(&recv_status, MPI_INT, &count);
                proc = recv_status.MPI_SOURCE;
                int recv_buffer[count];
                MPI_Recv(recv_buffer, count, MPI_INT, proc, 2468, MPI_COMM_WORLD, 
                        &recv_status);
                for (int i = 0; i < count; i++)
                {
                    global_col = recv_buffer[i];
                    idx = global_col - assumed_first;
                    recv_buffer[i] = assumed_col_procs[idx];
                }
                MPI_Send(recv_buffer, count, MPI_INT, proc, 3579, MPI_COMM_WORLD);
            }
            MPI_Testall(n_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, 2468, MPI_COMM_WORLD, &msg_avail, &recv_status);
        if (msg_avail)
        {
            MPI_Get_count(&recv_status, MPI_INT, &count);
            proc = recv_status.MPI_SOURCE;
            int recv_buffer[count];
            MPI_Recv(recv_buffer, count, MPI_INT, proc, 2468, MPI_COMM_WORLD, 
                    &recv_status);
            for (int i = 0; i < count; i++)
            {
                global_col = recv_buffer[i];
                idx = global_col - assumed_first;
                recv_buffer[i] = assumed_col_procs[idx];
            }
            MPI_Send(recv_buffer, count, MPI_INT, proc, 3579, MPI_COMM_WORLD);
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }

    if (n_sends)
    {
        MPI_Waitall(n_sends, recv_requests.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<int> send_sizes(num_procs);
    for (int i = 0; i < num_sends; i++)
    {
        send_sizes[send_procs[i]] = send_ptr[i];
    }

    for (int i = 0; i < off_proc_num_cols; i++)
    {
        global_col = off_proc_column_map[i];
        proc = global_col / assumed_part;
        if (proc == rank)
        {
            idx = global_col - assumed_first;
            off_proc_col_to_proc[i] = assumed_col_procs[idx];
        }
        else
        {
            idx = send_sizes[proc]++;
            off_proc_col_to_proc[i] = recv_indices[idx];
        }
    }
}

/**************************************************************
*****  Find Col To Proc  
**************************************************************
***** Will determine the process that stores corresponding
***** vector values for each column of the off_process 
***** portion of the par_matrix.
***** Note: this method no longer uses an Allreduce to scale
***** more efficiently to a large number of processes.
*****
***** Parameters
***** -------------
***** first_local_col : int 
*****    First column local to rank
***** global_num_cols : int 
*****    Total number of columns, partitioned across all ranks
***** local_num_cols : int 
*****    Number of columns local to rank
***** off_proc_column_map : std::vector<int>& 
*****    List of global columns in off_proc portion of par_matrix
***** off_proc_col_to_proc : std::vector<int>& 
*****    Will return with processes corresponding to each
*****    column in off_process block of matrix
**************************************************************/
void CommPkg::form_col_to_proc(const int first_local_col, 
        const int global_num_cols,
        const int local_num_cols, 
        const std::vector<int>& off_proc_column_map,
        std::vector<int>& off_proc_col_to_proc)
{            
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> col_starts;
    std::vector<int> col_start_procs;
    int last_local_col = first_local_col + local_num_cols;
    int assumed_part = find_proc_col_starts(first_local_col, last_local_col,
        global_num_cols, col_starts, col_start_procs);

    int off_proc_num_cols = off_proc_column_map.size();
    int col_start_size = col_starts.size();
    int prev_proc;
    int num_sends = 0;
    int proc, start, end;
    int col;
    std::vector<int> send_procs;
    std::vector<int> send_proc_starts;
    std::vector<MPI_Request> send_requests;

    if (off_proc_num_cols)
    {
        off_proc_col_to_proc.resize(off_proc_num_cols);
        prev_proc = off_proc_column_map[0] / assumed_part;
        send_procs.push_back(prev_proc);
        send_proc_starts.push_back(0);
        for (int i = 1; i < off_proc_num_cols; i++)
        {
            proc = off_proc_column_map[i] / assumed_part;
            if (proc != prev_proc)
            {
                send_procs.push_back(proc);
                send_proc_starts.push_back(i);
                prev_proc = proc;
            }
        }
        send_proc_starts.push_back(off_proc_num_cols);

        num_sends = send_procs.size();
        send_requests.resize(num_sends);
        for (int i = 0; i < num_sends; i++)
        {
            proc = send_procs[i];
            start = send_proc_starts[i];
            end = send_proc_starts[i+1];

            if (proc == rank)
            {
                int k = 0;
                for (int j = start; j < end; j++)
                {
                    col = off_proc_column_map[j];
                    while (k + 1 < col_start_size && 
                            col >= col_starts[k+1])
                    {
                        k++;
                    }
                    proc = col_start_procs[k];
                    off_proc_col_to_proc[j] = proc;
                }
                send_requests[i] = MPI_REQUEST_NULL;
            }
            else
            {
                MPI_Issend(&(off_proc_column_map[start]), end - start, MPI_INT,
                        proc, 9753, MPI_COMM_WORLD, &(send_requests[i]));
            }
        }
    }

    std::vector<int> sendbuf_procs;
    std::vector<int> sendbuf_starts;
    std::vector<int> sendbuf;
    int finished, msg_avail;
    int count;
    MPI_Request barrier_request;
    MPI_Status status;

    if (num_sends)
    {
        MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, 9753, MPI_COMM_WORLD, &msg_avail, &status);
            if (msg_avail)
            {
                MPI_Get_count(&status, MPI_INT, &count);
                proc = status.MPI_SOURCE;
                int recvbuf[count];
                MPI_Recv(recvbuf, count, MPI_INT, proc, 9753, 
                        MPI_COMM_WORLD, &status);
                sendbuf_procs.push_back(proc);
                sendbuf_starts.push_back(sendbuf.size());
                int k = 0;
                for (int i = 0; i < count; i++)
                {
                    col = recvbuf[i];
                    while (k + 1 < col_start_size &&
                            col >= col_starts[k + 1])
                    {
                        k++;
                    }
                    proc = col_start_procs[k];
                    sendbuf.push_back(proc);    
                }
            }
            MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
        }
    }
    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
    MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    while (!finished)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, 9753, MPI_COMM_WORLD, &msg_avail, &status);
        if (msg_avail)
        {
            MPI_Get_count(&status, MPI_INT, &count);
            proc = status.MPI_SOURCE;
            int recvbuf[count];
            MPI_Recv(recvbuf, count, MPI_INT, proc, 9753, 
                    MPI_COMM_WORLD, &status);
            sendbuf_procs.push_back(proc);
            sendbuf_starts.push_back(sendbuf.size());
            int k = 0;
            for (int i = 0; i < count; i++)
            {
                col = recvbuf[i];
                while (k + 1 < col_start_size &&
                        col >= col_starts[k + 1])
                {
                    k++;
                }
                proc = col_start_procs[k];
                sendbuf.push_back(proc);
            }
        }
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
    }
    sendbuf_starts.push_back(sendbuf.size());

    int n_sendbuf = sendbuf_procs.size();
    std::vector<MPI_Request> sendbuf_requests(n_sendbuf);
    for (int i = 0; i < n_sendbuf; i++)
    {
        int proc = sendbuf_procs[i];
        int start = sendbuf_starts[i];
        int end = sendbuf_starts[i+1];
        MPI_Isend(&(sendbuf[start]), end-start, MPI_INT, proc,
                8642, MPI_COMM_WORLD, &(sendbuf_requests[i]));
    }
        
    for (int i = 0; i < num_sends; i++)
    {
        int proc = send_procs[i];
        if (proc == rank) 
            continue;
        int start = send_proc_starts[i];
        int end = send_proc_starts[i+1];
        MPI_Irecv(&(off_proc_col_to_proc[start]), end - start, MPI_INT, proc,
                8642, MPI_COMM_WORLD, &(send_requests[i]));
    }

    MPI_Waitall(n_sendbuf, sendbuf_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(num_sends, send_requests.data(), MPI_STATUSES_IGNORE);

}

/**************************************************************
*****  Find Proc Col Starts 
**************************************************************
***** Assume each process holds and equal number of columns.
***** Hold col_starts and col_start_procs corresponding to 
***** the columns that are assumed to be stored locally.
*****
***** Parameters
***** -------------
***** global_num_cols : const int 
*****    Total number of columns, partitioned across all ranks
***** on_proc_column_map : const std::vector<int>& 
*****    Maps local columns to global indices.  Columns
*****    are not assumed to be partitioned across
*****    processes in any order.
***** assumed_col_procs : std::vector<int>& 
*****    Process holding each column assumed on rank
**************************************************************/
int CommPkg::find_proc_col_starts(const int global_num_cols, 
        const std::vector<int>& on_proc_column_map, 
        std::vector<int>& assumed_col_procs)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Delcare variables
    int part;
    int first, last;
    int start, end;
    int proc;
    int ctr, size;
    int num_assumed_cols;
    int count;
    int num_sends, size_recvd;
    int global_col, local_col;
    int local_num_cols = on_proc_column_map.size();
    std::vector<int> send_procs;
    std::vector<int> send_ptr;
    std::vector<int> send_indices;
    std::vector<MPI_Request> send_requests;
    MPI_Status recv_status;

    // Find assumed partition (cols per proc) and extra cols
    // which are assumed to be placed one per proc on first
    // "extra" num procs
    part = global_num_cols / num_procs;
    if (global_num_cols % num_procs) part++;

    // Find assumed first and last col on rank
    first = rank * part;
    last = (rank + 1) * part;
    if (last > global_num_cols) 
    {
        last = global_num_cols;
    }
    num_assumed_cols = last - first;
    assumed_col_procs.resize(num_assumed_cols);

    num_sends = form_send_data(part, on_proc_column_map, send_procs, 
            send_ptr, send_indices);
    if (num_sends)
    {
        send_requests.resize(num_sends);
    }

    // Send global column indices to process assumed to hold them
    size_recvd = 0;
    int n_sends = 0;
    for (int i = 0; i < num_sends; i++)
    {
        proc = send_procs[i];
        start = send_ptr[i];
        end = send_ptr[i+1];
        if (proc == rank)
        {
            for (int j = start; j < end; j++)
            {
                global_col = send_indices[j];
                local_col = global_col - first;
                assumed_col_procs[local_col] = proc;
            }
            size_recvd += (end - start);
        }
        else
        {
            MPI_Isend(&(send_indices[start]), end - start, MPI_INT, proc, 7654, 
                    MPI_COMM_WORLD, &(send_requests[n_sends++]));
        }
    }

    while (size_recvd < num_assumed_cols)
    {
        MPI_Probe(MPI_ANY_SOURCE, 7654, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, MPI_INT, &count);
        proc = recv_status.MPI_SOURCE;
        int recv_buffer[count];
        MPI_Recv(recv_buffer, count, MPI_INT, proc, 7654, MPI_COMM_WORLD, &recv_status);
        for (int i = 0; i < count; i++)
        {
            global_col = recv_buffer[i];
            local_col = global_col - first;
            assumed_col_procs[local_col] = proc;
        }
        size_recvd += count;
    }

    MPI_Waitall(n_sends, send_requests.data(), MPI_STATUSES_IGNORE);

    return part;
}

/**************************************************************
*****  Find Proc Col Starts 
**************************************************************
***** Assume each process holds and equal number of columns.
***** Hold col_starts and col_start_procs corresponding to 
***** the columns that are assumed to be stored locally.
*****
***** Parameters
***** -------------
***** first_local_col : int 
*****    First column local to rank
***** last_local_col : int 
*****    First column local to rank + 1
***** global_num_cols : int 
*****    Total number of columns, partitioned across all ranks
***** col_starts : std::vector<int>& 
*****    Will return with actual partition of all columns 
*****    that are assumed to be local to rank
***** col_start_procs : std::vector<int>& 
*****    Process holding each portion of col_starts
**************************************************************/
int CommPkg::find_proc_col_starts(const int first_local_col, 
        const int last_local_col,
        const int global_num_cols, 
        std::vector<int>& col_starts, 
        std::vector<int>& col_start_procs)
{
    // Get MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Delcare variables
    int part;
    int first, last;
    int proc;
    int num_procs_extra;
    int ctr;
    int tmp;
    int recvbuf;
    std::vector<int> send_buffer;
    std::vector<MPI_Request> send_requests;
    MPI_Status status;

    // Find assumed partition (cols per proc) and extra cols
    // which are assumed to be placed one per proc on first
    // "extra" num procs
    part = global_num_cols / num_procs;
    if (global_num_cols % num_procs) part++;

    // Find assumed first and last col on rank
    first = rank * part;
    last = (rank + 1) * part;
        
    // If last_local_col != last, exchange data with neighbors
    // Send to proc that is assumed to hold my last local col
    // and recv from all procs of which i hold their assumed last local cols
    proc = (last_local_col-1) / part; // first col of rank + 1
    num_procs_extra = proc - rank; 
    ctr = 0;
    if (num_procs_extra > 0)
    {
        send_buffer.resize(num_procs_extra);
        send_requests.resize(num_procs_extra);
        for (int i = rank + 1; i <= proc; i++)
        {
            send_buffer[ctr] = first_local_col;
            MPI_Isend(&(send_buffer[ctr]), 1, MPI_INT, i, 2345, 
                    MPI_COMM_WORLD, &(send_requests[ctr]));
            ctr++;
        }
    }
    tmp = first_local_col;
    proc = rank - 1;
    col_starts.push_back(tmp);
    while (first < tmp)
    {
        MPI_Recv(&recvbuf, 1, MPI_INT, proc, 2345, MPI_COMM_WORLD, &status);
        tmp = recvbuf;
        col_starts.push_back(tmp);
        proc--;
    }
    if (ctr)
    {
        MPI_Waitall(ctr, send_requests.data(), MPI_STATUSES_IGNORE);
    }

    // Reverse order of col_starts (lowest proc first)
    std::reverse(col_starts.begin(), col_starts.end());
    for (int i = col_starts.size() - 1; i >= 0; i--)
    {
        col_start_procs.push_back(rank - i);
    }

    // If first_local_col != first, exchange data with neighbors
    // Send to proc that is assumed to hold my first local col
    // and recv from all procs of which i hold their assumed first local cols
    proc = first_local_col / part;
    num_procs_extra = rank - proc;
    ctr = 0;
    if (num_procs_extra > 0)
    {
        send_buffer.resize(num_procs_extra);
        send_requests.resize(num_procs_extra);
        for (int i = proc; i < rank; i++)
        {
            send_buffer[ctr] = last_local_col;
            MPI_Isend(&(send_buffer[ctr]), 1, MPI_INT, i, 2345,
                    MPI_COMM_WORLD, &(send_requests[ctr]));
            ctr++;
        }
    }
    tmp = last_local_col;
    proc = rank + 1;
    col_starts.push_back(last_local_col);
    col_start_procs.push_back(proc);
    while (last > tmp && proc < num_procs)
    {
        MPI_Recv(&recvbuf, 1, MPI_INT, proc, 2345, MPI_COMM_WORLD, &status);
        tmp = recvbuf;
        col_starts.push_back(recvbuf);
        col_start_procs.push_back(++proc);
    }
    if (ctr)
    {
        MPI_Waitall(ctr, send_requests.data(), MPI_STATUSES_IGNORE);
    }

    return part;
}

void ParComm::communicate(data_t* values, MPI_Comm comm)
{
    init_comm(values, comm);
    complete_comm();
}

void ParComm::init_comm(data_t* values, MPI_Comm comm)
{
    if (send_data->num_msgs)
    {
        int send_start;
        int send_end;
        int proc;

        std::vector<int>& procs = send_data->procs;
        std::vector<int>& indptr = send_data->indptr;
        std::vector<int>& indices = send_data->indices;
        double* buffer = send_data->buffer.data();
        MPI_Request* requests = send_data->requests;

        // Add local data to buffer, and send to appropriate procs
        for (int i = 0; i < send_data->num_msgs; i++)
        {
            proc = procs[i];
            send_start = indptr[i];
            send_end = indptr[i+1];
            for (int j = send_start; j < send_end; j++)
            {
                buffer[j] = values[indices[j]];
            }
            MPI_Isend(&(buffer[send_start]), send_end - send_start,
                    MPI_DATA_T, proc, 0, comm, &(requests[i]));
        }
    }

    if (recv_data->num_msgs)
    {
        int recv_start;
        int recv_end;
        int proc;

        std::vector<int>& procs = recv_data->procs;
        std::vector<int>& indptr = recv_data->indptr;
        double* buffer = recv_data->buffer.data();
        MPI_Request* requests = recv_data->requests;

        for (int i = 0; i < recv_data->num_msgs; i++)
        {
            proc = procs[i];
            recv_start = indptr[i];
            recv_end = indptr[i+1];
            MPI_Irecv(&(buffer[recv_start]), recv_end - recv_start, 
                    MPI_DATA_T, proc, 0, comm, &(requests[i]));
        }
    }
}

void ParComm::complete_comm()
{
    if (send_data->num_msgs)
    {
        MPI_Waitall(send_data->num_msgs, send_data->requests, MPI_STATUS_IGNORE);
    }

    if (recv_data->num_msgs)
    {
        MPI_Waitall(recv_data->num_msgs, recv_data->requests, MPI_STATUS_IGNORE);
    }
}

void TAPComm::communicate(data_t* values, MPI_Comm comm)
{
    init_comm(values, comm);
    complete_comm();
}

void TAPComm::init_comm(data_t* values, MPI_Comm comm)
{
    // Messages with origin and final destination on node
    local_L_par_comm->init_comm(values, local_comm);
    local_L_par_comm->complete_comm();

    // Initial redistribution among node
    local_S_par_comm->init_comm(values, local_comm);
    local_S_par_comm->complete_comm();
    data_t* S_vals = local_S_par_comm->recv_data->buffer.data();

    // Begin inter-node communication 
    global_par_comm->init_comm(S_vals, comm);
}

void TAPComm::complete_comm()
{
    // Complete inter-node communication
    global_par_comm->complete_comm();
    data_t* G_vals = global_par_comm->recv_data->buffer.data();

    // Redistributing recvd inter-node values
    local_R_par_comm->init_comm(G_vals, local_comm);
    local_R_par_comm->complete_comm();
    Vector& R_recv = local_R_par_comm->recv_data->buffer;

    Vector& L_recv = local_L_par_comm->recv_data->buffer;

    // Add values from L_recv and R_recv to appropriate positions in 
    // Vector recv
    int idx;
    for (int i = 0; i < R_recv.size; i++)
    {
        idx = R_to_orig[i];
        recv_buffer.values[idx] = R_recv.values[i];
    }

    for (int i = 0; i < L_recv.size; i++)
    {
        idx = L_to_orig[i];
        recv_buffer.values[idx] = L_recv.values[i];
    }
}

CSRMatrix* ParComm::communication_helper(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm, CommData* send_comm, CommData* recv_comm)
{
    // Number of rows in recv_mat = size_recvs
    // Don't know number of columns, but does not matter (CSR)
    CSRMatrix* recv_mat = new CSRMatrix(recv_comm->size_msgs, -1);

    int start, end, proc;
    int row, row_size;
    int send_mat_size;
    int ctr, prev_ctr;
    int row_start, row_end;
    int start_idx, end_idx;

    // Calculate nnz/row for each row to be sent to a proc
    std::vector<int> send_row_buffer;
    std::vector<int> recv_row_buffer;
    if (send_comm->num_msgs)
    {
        send_row_buffer.resize(send_comm->size_msgs);
    }
    if (recv_comm->num_msgs)
    {
        recv_row_buffer.resize(recv_comm->size_msgs);
    }

    // Send nnz/row for each row to be communicated
    send_mat_size = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        proc = send_comm->procs[i];
        for (int j = start; j < end; j++)
        {
            row = send_comm->indices[j];
            row_size = rowptr[row+1] - rowptr[row];
            send_row_buffer[j] = row_size;
            send_mat_size += row_size;
        }

        MPI_Isend(&(send_row_buffer[start]), end - start, MPI_INT, proc,
                key, comm, &(send_comm->requests[i]));
    }

    // Recv nnz/row for each row to be received
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        proc = recv_comm->procs[i];
        
        MPI_Irecv(&(recv_row_buffer[start]), end - start, MPI_INT, proc,
                key, comm, &(recv_comm->requests[i]));
    }

    // Wait for communication to complete
    if (recv_comm->num_msgs)
    {
        MPI_Waitall(recv_comm->num_msgs, recv_comm->requests, MPI_STATUS_IGNORE);
    }
    if (send_comm->num_msgs)
    {
        MPI_Waitall(send_comm->num_msgs, send_comm->requests, MPI_STATUS_IGNORE);
    }

    // Allocate Matrix Space
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < recv_comm->size_msgs; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + recv_row_buffer[i];
    }
    recv_mat->nnz = recv_mat->idx1[recv_comm->size_msgs];

    if (recv_mat->nnz)
    {
        recv_mat->idx2.resize(recv_mat->nnz);
        recv_mat->vals.resize(recv_mat->nnz);
    }

    // Create PairData for sends and recvs (pair of int
    // and double: col idx and value)
    struct PairData 
    {
        double val;
        int index;
    };
    std::vector<PairData> send_buffer;
    std::vector<PairData> recv_buffer;
    if (send_mat_size)
    {
        send_buffer.resize(send_mat_size);
    }
    if (recv_mat->nnz)
    {
        recv_buffer.resize(recv_mat->nnz);
    }

    // Send pair_data for each row using MPI_DOUBLE_INT
    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < send_comm->num_msgs; i++)
    {
        start = send_comm->indptr[i];
        end = send_comm->indptr[i+1];
        proc = send_comm->procs[i];

        for (int j = start; j < end; j++)
        {
            row = send_comm->indices[j];
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                send_buffer[ctr].val = values[k];
                send_buffer[ctr++].index = col_indices[k];
            }
        }

        if (ctr - prev_ctr)
        {
            MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, MPI_DOUBLE_INT, proc, 
                    key, comm, &(send_comm->requests[i]));
            prev_ctr = ctr;
        }
        else
        {
            send_comm->requests[i] = MPI_REQUEST_NULL;
        }
    }

    // Recv pair_data corresponding to each off_proc column and add it to
    // correct location in matrix
    for (int i = 0; i < recv_comm->num_msgs; i++)
    {
        start = recv_comm->indptr[i];
        end = recv_comm->indptr[i+1];
        proc = recv_comm->procs[i];

        start_idx = recv_mat->idx1[start];
        end_idx = recv_mat->idx1[end];

        if (end_idx - start_idx)
        {
            MPI_Irecv(&(recv_buffer[start_idx]), end_idx - start_idx, MPI_DOUBLE_INT,
                    proc, key, comm, &(recv_comm->requests[i]));
        }
        else
        {
            recv_comm->requests[i] = MPI_REQUEST_NULL;
        }
    }

    if (recv_comm->num_msgs)
    {
        MPI_Waitall(recv_comm->num_msgs, recv_comm->requests, MPI_STATUS_IGNORE);
    }
    if (send_comm->num_msgs)
    {
        MPI_Waitall(send_comm->num_msgs, send_comm->requests, MPI_STATUS_IGNORE);
    }

    // Add recvd values to matrix
    for (int i = 0; i < recv_mat->nnz; i++)
    {
        recv_mat->idx2[i] = recv_buffer[i].index;
        recv_mat->vals[i] = recv_buffer[i].val;
    }

    return recv_mat;
}


CSRMatrix* TAPComm::communicate(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{   
    int ctr, idx;
    int start, end;

    CSRMatrix* S_mat = local_S_par_comm->communicate(rowptr, col_indices, values,
            local_comm);
    CSRMatrix* G_mat = global_par_comm->communicate(S_mat->idx1, S_mat->idx2,
            S_mat->vals, comm);
    delete S_mat;

    CSRMatrix* R_mat = local_R_par_comm->communicate(G_mat->idx1, G_mat->idx2,
            G_mat->vals, local_comm);
    delete G_mat;

    CSRMatrix* L_mat = local_L_par_comm->communicate(rowptr, col_indices, values,
            local_comm);

    // Create recv_mat (combo of L_mat and R_mat)
    CSRMatrix* recv_mat = new CSRMatrix(L_mat->n_rows + R_mat->n_rows, -1);
    int nnz = L_mat->nnz + R_mat->nnz;
    if (nnz)
    {
        recv_mat->idx2.resize(nnz);
        recv_mat->vals.resize(nnz);
    }

    ctr = 0;
    recv_mat->idx1[0] = ctr;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        if (orig_to_R[i] >= 0)
        {
            idx = orig_to_R[i];
            start = R_mat->idx1[idx];
            end = R_mat->idx1[idx+1];
            for (int j = start; j < end; j++)
            {
                recv_mat->idx2[ctr] = R_mat->idx2[j];
                recv_mat->vals[ctr++] = R_mat->vals[j];
            }
        }
        else
        {
            idx = orig_to_L[i];
            start = L_mat->idx1[idx];
            end = L_mat->idx1[idx+1];
            for (int j = start; j < end; j++)
            {
                recv_mat->idx2[ctr] = L_mat->idx2[j];
                recv_mat->vals[ctr++] = L_mat->vals[j];
            }
        }
        recv_mat->idx1[i+1] = ctr;
    }
    
    delete R_mat;
    delete L_mat;

    return recv_mat;
}


CSRMatrix* ParComm::communicate(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{
    return communication_helper(rowptr, col_indices, values, comm,
            send_data, recv_data);
}

CSRMatrix* ParComm::communicate_T(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{
    return communication_helper(rowptr, col_indices, values, comm,
            recv_data, send_data);
}
    

// TODO -- this needs fixed (how to do transpose TAP comm)??
CSRMatrix* TAPComm::communicate_T(std::vector<int>& rowptr, 
        std::vector<int>& col_indices, std::vector<double>& values,
        MPI_Comm comm)
{   
    int ctr, idx;
    int start, end;

    CSRMatrix* S_mat = local_S_par_comm->communicate(rowptr, col_indices, values,
            local_comm);
    CSRMatrix* G_mat = global_par_comm->communicate(S_mat->idx1, S_mat->idx2,
            S_mat->vals, comm);
    delete S_mat;

    CSRMatrix* R_mat = local_R_par_comm->communicate(G_mat->idx1, G_mat->idx2,
            G_mat->vals, local_comm);
    delete G_mat;

    CSRMatrix* L_mat = local_L_par_comm->communicate(rowptr, col_indices, values,
            local_comm);

    // Create recv_mat (combo of L_mat and R_mat)
    CSRMatrix* recv_mat = new CSRMatrix(L_mat->n_rows + R_mat->n_rows, -1);
    int nnz = L_mat->nnz + R_mat->nnz;
    if (nnz)
    {
        recv_mat->idx2.resize(nnz);
        recv_mat->vals.resize(nnz);
    }

    ctr = 0;
    recv_mat->idx1[0] = ctr;
    for (int i = 0; i < recv_mat->n_rows; i++)
    {
        if (orig_to_R[i] >= 0)
        {
            idx = orig_to_R[i];
            start = R_mat->idx1[idx];
            end = R_mat->idx1[idx+1];
            for (int j = start; j < end; j++)
            {
                recv_mat->idx2[ctr] = R_mat->idx2[j];
                recv_mat->vals[ctr++] = R_mat->vals[j];
            }
        }
        else
        {
            idx = orig_to_L[i];
            start = L_mat->idx1[idx];
            end = L_mat->idx1[idx+1];
            for (int j = start; j < end; j++)
            {
                recv_mat->idx2[ctr] = L_mat->idx2[j];
                recv_mat->vals[ctr++] = L_mat->vals[j];
            }
        }
        recv_mat->idx1[i+1] = ctr;
    }
    
    delete R_mat;
    delete L_mat;

    return recv_mat;
}



