// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include <math.h>

#include "matrix.hpp"
#include "par_vector.hpp"
#include "comm_data.hpp"
#include <map>

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
        int* global_col_starts;
        int* proc_sends;
        int* proc_recvs;
        int* tmp_send_buffer;
        int* col_to_proc;
        int n_recvd, ctr;
        int send_start, send_end;
        int num_sends;
        int proc, prev_proc;
        int global_col, global_row, local_row;
        int count;
        int tag = 2345;  // TODO -- switch this to key?
        int off_proc_num_cols = off_proc_column_map.size();
        MPI_Status recv_status;

        // Allocate space for processes I send to / recv from
        proc_sends = new int[num_procs]();
        proc_recvs = new int[num_procs];

        // Create global_col_starts (first global col on each proc)
        global_col_starts = new int[num_procs];
        MPI_Allgather(&first_local_col, 1, MPI_INT, 
                global_col_starts, 1, MPI_INT, MPI_COMM_WORLD);

        // Map columns of off_proc to the processes on 
        // which associated vector values are stored
        col_to_proc = new int[off_proc_num_cols];
        proc = 0;
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            global_col = off_proc_column_map[i];
            while (proc + 1 < num_procs && 
                    global_col >= global_col_starts[proc + 1])
                proc++;
            col_to_proc[i] = proc;
        }

        // Determine processes columns are received from,
        // and adds corresponding messages to recv data.
        // Assumes columns are partitioned across processes
        // in contiguous blocks, and are sorted
        prev_proc = col_to_proc[0];
        proc_sends[prev_proc] = 1;
        int prev_idx = 0;
        for (int i = 1; i < off_proc_num_cols; i++)
        {
            proc = col_to_proc[i];
            if (proc != prev_proc)
            {
                recv_data->add_msg(prev_proc, i - prev_idx);
                prev_proc = proc;
                prev_idx = i;
                proc_sends[proc] = 1;
            }
        }
        recv_data->add_msg(prev_proc, off_proc_num_cols - prev_idx);
        recv_data->finalize();

        // Find the number of processes that must recv from me
        MPI_Allreduce(proc_sends, proc_recvs, num_procs, MPI_INT, MPI_SUM, comm);
        num_sends = proc_recvs[rank];

        // For each process I recv from, send the global column indices
        // for which I must recv corresponding rows 
        if (recv_data->size_msgs) tmp_send_buffer = new int[recv_data->size_msgs];
        for (int i = 0; i < recv_data->num_msgs; i++)
        {
            proc = recv_data->procs[i];
            send_start = recv_data->indptr[i];
            send_end = recv_data->indptr[i+1];
            for (int j = send_start; j < send_end; j++)
            {
                tmp_send_buffer[j] = off_proc_column_map[j];
            }
            MPI_Isend(&(tmp_send_buffer[send_start]), send_end - send_start, MPI_INT, 
                    proc, tag, comm, &(recv_data->requests[i]));
        }

        // Determine which processes to which I send messages,
        // and what vector indices to send to each.
        // Receive any messages, regardless of source (which is unknown)
        n_recvd = 0;
        ctr = 0;
        while (n_recvd < num_sends)
        {
            // Wait until message is in buffer
            MPI_Probe(MPI_ANY_SOURCE, tag, comm, &recv_status);

            // Find size of message
            MPI_Get_count(&recv_status, MPI_INT, &count);

            // Find process sending message
            proc = recv_status.MPI_SOURCE;

            // Recv first message in buffer
            int recvbuf[count];
            MPI_Recv(recvbuf, count, MPI_INT, proc, tag, comm, 
                    &recv_status);
    
            // Add destination process to send_procs as this process
            // will send vector values to me
            for (int i = 0; i < count; i++)
                recvbuf[i] -= first_local_row;
            send_data->add_msg(proc, count, recvbuf);

            n_recvd++;
        }
        send_data->finalize();
        
        // Wait for sends to complete
        if (recv_data->num_msgs) 
            MPI_Waitall(recv_data->num_msgs, recv_data->requests, MPI_STATUS_IGNORE);

        // Delete variables
        if (recv_data->size_msgs) 
            delete[] tmp_send_buffer;
        delete[] col_to_proc;
        delete[] global_col_starts;
        delete[] proc_sends;
        delete[] proc_recvs;
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

    int key;
    CommData* send_data;
    CommData* recv_data;

};
}
#endif
