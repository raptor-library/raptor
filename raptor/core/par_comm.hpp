// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include <math.h>

#include "matrix.hpp"
#include "par_vector.hpp"
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
    ***** global_col_starts : std::vector<index_t>&
    *****    Describes the partition of the parallel matrix
    **************************************************************/
    void init_col_to_proc(const MPI_Comm comm_mat, const index_t num_cols, std::map<index_t, index_t>& global_to_local, std::vector<index_t>& global_col_starts);

    /**************************************************************
    *****   ParComm Init Communicator Recvs
    **************************************************************
    ***** Initializes the recv data for communication.  Sets the
    *****
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
    ***** global_col_starts : std::vector<index_t>&
    *****    Describes the partition of the parallel matrix
    **************************************************************/
    void init_comm_recvs(const MPI_Comm comm_mat, const index_t num_cols, std::map<index_t, index_t>& global_to_local);
    void init_comm_sends_unsym(const MPI_Comm comm_mat, std::vector<index_t>& map_to_global, std::vector<index_t>& global_col_starts);

    void init_sends(const data_t* x_data, MPI_Comm comm = MPI_COMM_WORLD);
    void init_recvs(MPI_Comm comm = MPI_COMM_WORLD);
    void complete_sends();
    void complete_recvs();
    int init_mat_comm(int* row_starts, csr_data* tmp_data, MPI_Datatype csr_type, MPI_Comm comm = MPI_COMM_WORLD);
    void clean_mat_comm();

    /**************************************************************
    *****   ParComm Class Constructor
    **************************************************************
    ***** Initializes an empty ParComm, setting send and recv
    ***** sizes to 0
    **************************************************************/
    ParComm(int _key = 0)
    {
        num_sends = 0;
        size_sends = 0;
        num_recvs = 0;
        size_recvs = 0;
        key = _key;
    }

    /**************************************************************
    *****   ParComm Class Constructor
    **************************************************************
    ***** Initializes a ParComm object based on the offd Matrix
    *****
    ***** Parameters
    ***** -------------
    ***** offd : Matrix*
    *****    Matrix holding local off-diagonal block
    ***** map_to_global : std::vector<index_t>&
    *****    Maps local columns of offd block to global columns
    *****    of the parallel matrix
    ***** global_to_local : std::map<index_t, index_t>&
    *****    Maps global columns of the parallel matrix to local
    *****    columns of the off-diagonal block
    ***** global_col_starts : std::vector<index_t>& 
    *****    Describes the partition of the parallel matrix
    *****    across the processors
    ***** comm_mat : MPI_Comm
    *****    MPI Communicator containing all active processes
    *****    (all processes containing at least on row of the matrix)
    **************************************************************/
    ParComm(std::vector<index_t>& local_to_global, std::vector<index_t>& global_col_starts, int first_row, const MPI_Comm comm_mat = MPI_COMM_WORLD, int _key = 9999)
    {
        num_sends = 0;
        size_sends = 0;
        num_recvs = 0;
        size_recvs = 0;
        key = _key;

        // Get MPI Information
        int rank, num_procs;
        MPI_Comm_rank(comm_mat, &rank);
        MPI_Comm_size(comm_mat, &num_procs);

        // Declare communication variables
        index_t offd_num_cols = local_to_global.size();
        int* proc_sends;
        int* proc_recvs;
        int ctr;
        int n_recvd;
        int send_start, send_end;
        int proc, prev_proc;
        int global_col;
        int global_row, local_row;
        int tag = 2345;
        int count;
        MPI_Status recv_status;
        int* tmp_send_buffer;

        proc_sends = new int[num_procs]();
        proc_recvs = new int[num_procs];
        proc = 0;
        prev_proc = -1;
        for (int i = 0; i < offd_num_cols; i++)
        {
            global_col = local_to_global[i];
            while (proc + 1 < num_procs && 
                    global_col >= global_col_starts[proc + 1])
                proc++;
            if (proc != prev_proc)
            {
                recv_procs.push_back(proc);
                recv_col_starts.push_back(i);
                proc_sends[proc] = 1;
                prev_proc = proc;
            }
        }
        recv_col_starts.push_back(offd_num_cols);
        num_recvs = recv_procs.size();
        size_recvs = recv_col_starts[num_recvs];
        if (num_recvs)
        {
            recv_requests = new MPI_Request[num_recvs];
            recv_buffer = new data_t[size_recvs];
        }
        
        MPI_Allreduce(proc_sends, proc_recvs, num_procs, MPI_INT, MPI_SUM, comm_mat);
        num_sends = proc_recvs[rank];

        if (size_recvs) tmp_send_buffer = new int[size_recvs];
        for (int i = 0; i < num_recvs; i++)
        {
            proc = recv_procs[i];
            send_start = recv_col_starts[i];
            send_end = recv_col_starts[i+1];
            for (int j = send_start; j < send_end; j++)
            {
                tmp_send_buffer[j] = local_to_global[j];
            }
            MPI_Isend(&(tmp_send_buffer[send_start]), send_end - send_start, MPI_INT, 
                    proc, tag, comm_mat, &(recv_requests[i]));
        }

        n_recvd = 0;
        ctr = 0;
        send_row_starts.push_back(0);
        while (n_recvd < num_sends)
        {
            // Wait until message is in buffer
            MPI_Probe(MPI_ANY_SOURCE, tag, comm_mat, &recv_status);

            // Find size of message
            MPI_Get_count(&recv_status, MPI_INT, &count);
            proc = recv_status.MPI_SOURCE;

            // Recv first message in buffer
            int recvbuf[count];
            MPI_Recv(recvbuf, count, MPI_INT, proc, tag, comm_mat, 
                    &recv_status);
    
            send_row_starts.push_back(send_row_starts[ctr++] + count);
            send_procs.push_back(proc);

            for (int i = 0; i < count; i++)
            {
                global_row = recvbuf[i];
                local_row = global_row - first_row;
                send_row_indices.push_back(local_row);
            }
            n_recvd++;
        }
        size_sends = send_row_starts[num_sends];
        if (num_sends)
        {
            send_requests = new MPI_Request[num_sends];
            send_buffer = new data_t[size_sends];
        }

        if (num_recvs) MPI_Waitall(num_recvs, recv_requests, MPI_STATUS_IGNORE);

        if (size_recvs) delete[] tmp_send_buffer;
        delete[] proc_sends;
        delete[] proc_recvs;
    }

    /**************************************************************
    *****   ParComm Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    ~ParComm()
    {
        if (num_sends)
        {
            delete[] send_requests;
            delete[] send_buffer;
        }
        
        if (num_recvs)
        {
            delete[] recv_requests;
            delete[] recv_buffer;
        }
    };

    index_t num_sends;
    index_t num_recvs;
    index_t size_sends;
    index_t size_recvs;
    int key;
    std::vector<index_t> send_procs;
    std::vector<index_t> send_row_starts;
    std::vector<index_t> send_row_indices;
    std::vector<index_t> recv_procs;
    std::vector<index_t> recv_col_starts;
    std::vector<index_t> col_to_proc;
    MPI_Request* send_requests;
    MPI_Request* recv_requests; 
    data_t* send_buffer;
    data_t* recv_buffer;
    int* row_send_buffer;
    int* row_recv_buffer;
    int* csr_row_starts;
    csr_data* csr_send_buffer;
    csr_data* csr_recv_buffer;
};
}
#endif
