// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include "comm_data.hpp"

/**************************************************************
 *****   ParComm Class
 **************************************************************
 ***** This class constructs a parallel communicator, containing
 ***** which messages must be sent/recieved for matrix operations
 *****
 ***** Attributes
 ***** -------------
 ***** num_sends : index_t
 *****    Number of messages this process must send during 
 *****    matrix operations
 ***** num_recvs : index_t
 *****    Number of messages this process will recv during
 *****    matrix operations
 ***** size_sends : index_t 
 *****    Total number of elements this process sends in all
 *****    messages
 ***** size_recvs : index_t 
 *****    Total number of elements this process recvs from
 *****    all messages
 ***** send_procs : std::vector<index_t>
 *****    Distant processes messages are to be sent to
 ***** send_row_starts : std::vector<index_t>
 *****    Pointer to first position in send_row_indices
 *****    that a given process will send.
 ***** send_row_indices : std::vector<index_t> 
 *****    The indices of values that must be sent to each
 *****    process in send_procs
 ***** recv_procs : std::vector<index_t>
 *****    Distant processes messages are to be recvd from
 ***** recv_col_starts : std::vector<index_t>
 *****    Pointer to first column recvd from each process
 *****    in recv_procs
 ***** col_to_proc : std::vector<index_t>
 *****    Maps each local column in the off-diagonal block
 *****    to the process that holds corresponding data
 ***** 
 ***** Methods
 ***** -------
 ***** init_col_to_proc()
 *****    Initializes col_to_proc, adding the process that 
 *****    corresponds to each column in the off-diagonal block
 ***** init_comm_recvs()
 *****    Initializes the recv data (recv_procs and recv_col_starts)
 ***** init_comm_sends()
 *****    Initializes the send data, based on the previously
 *****    initialized recv_data
 **************************************************************/
namespace raptor
{
class ParComm
{
public:
    /**************************************************************
    *****   ParComm Class Constructor
    **************************************************************
    ***** Initializes an empty ParComm, setting send and recv
    ***** sizes to 0
    ***** _key : int (optional)
    *****    Tag to be used in MPI Communication (default 0)
    **************************************************************/
    ParComm(int _key = 0)
    {
        key = _key;
        send_data = new CommData();
        recv_data = new CommData();
    }

    /**************************************************************
    *****   ParComm Class Constructor
    **************************************************************
    ***** Initializes a ParComm object based on the off_proc Matrix
    *****
    ***** Parameters
    ***** -------------
    ***** off_proc_column_map : std::vector<index_t>&
    *****    Maps local off_proc columns indices to global
    ***** first_local_row : index_t
    *****    Global row index of first row local to process
    ***** first_local_col : index_t
    *****    Global row index of first column to fall in local block
    ***** _key : int (optional)
    *****    Tag to be used in MPI Communication (default 9999)
    **************************************************************/
    ParComm(std::vector<int>& off_proc_column_map,
            int first_local_row, 
            int first_local_col,
            int global_num_cols,
            int local_num_cols,
            int _key = 9999,
            MPI_Comm comm = MPI_COMM_WORLD)
    {
        // Get MPI Information
        int rank, num_procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_procs);

        // Initialize class variables
        key = _key;
        send_data = new CommData();
        recv_data = new CommData();

        // Declare communication variables
        int send_start, send_end;
        int proc, prev_proc;
        int count;
        int tag = 2345;  // TODO -- switch this to key?
        int off_proc_num_cols = off_proc_column_map.size();
        MPI_Status recv_status;

        std::vector<int> off_proc_col_to_proc(off_proc_num_cols);
        std::vector<int> tmp_send_buffer;

        form_col_to_proc(first_local_col, global_num_cols, local_num_cols, 
                off_proc_column_map, off_proc_col_to_proc);

        // Determine processes columns are received from,
        // and adds corresponding messages to recv data.
        // Assumes columns are partitioned across processes
        // in contiguous blocks, and are sorted
        prev_proc = off_proc_col_to_proc[0];
        int prev_idx = 0;
        for (int i = 1; i < off_proc_num_cols; i++)
        {
            proc = off_proc_col_to_proc[i];
            if (proc != prev_proc)
            {
                recv_data->add_msg(prev_proc, i - prev_idx);
                prev_proc = proc;
                prev_idx = i;
            }
        }
        recv_data->add_msg(prev_proc, off_proc_num_cols - prev_idx);
        recv_data->finalize();

        // For each process I recv from, send the global column indices
        // for which I must recv corresponding rows 
        if (recv_data->size_msgs)
        {
            tmp_send_buffer.resize(recv_data->size_msgs);
        }
        for (int i = 0; i < recv_data->num_msgs; i++)
        {
            proc = recv_data->procs[i];
            send_start = recv_data->indptr[i];
            send_end = recv_data->indptr[i+1];
            for (int j = send_start; j < send_end; j++)
            {
                tmp_send_buffer[j] = off_proc_column_map[j];
            }
            MPI_Issend(&(tmp_send_buffer[send_start]), send_end - send_start, MPI_INT, 
                    proc, tag, comm, &(recv_data->requests[i]));
        }

        // Determine which processes to which I send messages,
        // and what vector indices to send to each.
        // Receive any messages, regardless of source (which is unknown)
        int finished, msg_avail;
        MPI_Request barrier_request;
        MPI_Testall(recv_data->num_msgs, recv_data->requests, &finished,
                MPI_STATUSES_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &msg_avail, &recv_status);
            if (msg_avail)
            {
                MPI_Get_count(&recv_status, MPI_INT, &count);
                proc = recv_status.MPI_SOURCE;
                int recvbuf[count];
                MPI_Recv(recvbuf, count, MPI_INT, proc, tag, comm, &recv_status);
                for (int i = 0; i < count; i++)
                {
                    recvbuf[i] -= first_local_row;
                }
                send_data->add_msg(proc, count, recvbuf);
            }
            MPI_Testall(recv_data->num_msgs, recv_data->requests, &finished,
                    MPI_STATUSES_IGNORE);
        }
        MPI_Ibarrier(comm, &barrier_request);
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &msg_avail, &recv_status);
            if (msg_avail)
            {
                MPI_Get_count(&recv_status, MPI_INT, &count);
                proc = recv_status.MPI_SOURCE;
                int recvbuf[count];
                MPI_Recv(recvbuf, count, MPI_INT, proc, tag, comm, &recv_status);
                for (int i = 0; i < count; i++)
                {
                    recvbuf[i] -= first_local_row;
                }
                send_data->add_msg(proc, count, recvbuf);
            }
            MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
        }
        send_data->finalize();
    }

    ParComm(ParComm* comm)
    {
        send_data = new CommData(comm->send_data);
        recv_data = new CommData(comm->recv_data);
        key = comm->key;
    }

    /**************************************************************
    *****   ParComm Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    ~ParComm()
    {
        delete send_data;
        delete recv_data;
    };

    int find_proc_col_starts(int first_local_col, int last_local_col,
            int global_num_cols, std::vector<int>& col_starts, 
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
        proc = last_local_col / part - 1; // first col of rank + 1
        num_procs_extra = proc - rank; 
        ctr = 0;
        if (num_procs_extra > 0)
        {
            send_buffer.resize(num_procs_extra);
            send_requests.resize(num_procs_extra);
            for (int i = rank+1; i < proc; i++)
            {
                send_buffer[ctr] = first_local_col;
                MPI_Isend(&(send_buffer[ctr]), 1, MPI_INT, i, 2345, 
                        MPI_COMM_WORLD, &(send_requests[ctr]));
                ctr++;
            }
        }
        tmp = first_local_col;
        proc = rank - 1;
        col_starts.push_back(first_local_col);
        while (first < tmp && proc >= 0)
        {
            MPI_Recv(&recvbuf, 1, MPI_INT, proc, 2345, MPI_COMM_WORLD, &status);
            tmp = recvbuf;
            col_starts.push_back(first);
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

    void form_col_to_proc(int first_local_col, int global_num_cols,
            int local_num_cols, std::vector<int>& off_proc_column_map,
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

    int key;
    CommData* send_data;
    CommData* recv_data;

};
}
#endif
