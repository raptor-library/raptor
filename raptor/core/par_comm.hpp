// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include <math.h>

#include "matrix.hpp"
#include "array.hpp"
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
 ***** send_procs : Array<index_t>
 *****    Distant processes messages are to be sent to
 ***** send_row_starts : Array<index_t>
 *****    Pointer to first position in send_row_indices
 *****    that a given process will send.
 ***** send_row_indices : Array<index_t> 
 *****    The indices of values that must be sent to each
 *****    process in send_procs
 ***** recv_procs : Array<index_t>
 *****    Distant processes messages are to be recvd from
 ***** recv_col_starts : Array<index_t>
 *****    Pointer to first column recvd from each process
 *****    in recv_procs
 ***** col_to_proc : Array<index_t>
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
    ***** global_col_starts : Array<index_t>&
    *****    Describes the partition of the parallel matrix
    **************************************************************/
    void init_col_to_proc(const MPI_Comm comm_mat, const index_t num_cols, std::map<index_t, index_t>& global_to_local, Array<index_t>& global_col_starts);

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
    ***** global_col_starts : Array<index_t>&
    *****    Describes the partition of the parallel matrix
    **************************************************************/
    void init_comm_recvs(const MPI_Comm comm_mat, const index_t num_cols, std::map<index_t, index_t>& global_to_local);
    void init_comm_sends_unsym(const MPI_Comm comm_mat, Array<index_t>& map_to_global, Array<index_t>& global_col_starts);

    /**************************************************************
    *****   ParComm Class Constructor
    **************************************************************
    ***** Initializes an empty ParComm, setting send and recv
    ***** sizes to 0
    **************************************************************/
    ParComm()
    {
        num_sends = 0;
        size_sends = 0;
        num_recvs = 0;
        size_recvs = 0;
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
    ***** map_to_global : Array<index_t>&
    *****    Maps local columns of offd block to global columns
    *****    of the parallel matrix
    ***** global_to_local : std::map<index_t, index_t>&
    *****    Maps global columns of the parallel matrix to local
    *****    columns of the off-diagonal block
    ***** global_col_starts : Array<index_t>& 
    *****    Describes the partition of the parallel matrix
    *****    across the processors
    ***** comm_mat : MPI_Comm
    *****    MPI Communicator containing all active processes
    *****    (all processes containing at least on row of the matrix)
    **************************************************************/
    ParComm(const Matrix* offd, Array<index_t>& map_to_global, std::map<index_t, index_t>& global_to_local, Array<index_t>& global_col_starts, const MPI_Comm comm_mat)
    {
        num_sends = 0;
        size_sends = 0;
        num_recvs = 0;
        size_recvs = 0;

        // Get MPI Information
        int rank, num_procs;
        MPI_Comm_rank(comm_mat, &rank);
        MPI_Comm_size(comm_mat, &num_procs);

        // Declare communication variables
        index_t offd_num_cols = map_to_global.size();

        // For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map recvIndices
        if (offd_num_cols)
        {
            // Create map from columns to processors they lie on
            init_col_to_proc(comm_mat, offd_num_cols, global_to_local, global_col_starts);
            
            // Init recvs
            init_comm_recvs(comm_mat, offd_num_cols, global_to_local);
        }

        // Add processors needing to send to, and what to send to each
        init_comm_sends_unsym(comm_mat, map_to_global, global_col_starts);

        if (num_sends)
        {
            send_requests = new MPI_Request [num_sends];
            send_buffer = new data_t[size_sends];
        }

        if (num_recvs)
        {
            recv_requests = new MPI_Request [num_recvs];
            recv_buffer = new data_t [size_recvs];
        }
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

#ifdef WITH_AMPI
    void pup(PUP::er &p)
    {
        // Basic Primitives
        p | num_sends;
        p | num_recvs;
        p | size_sends;
        p | size_recvs;

        // Custom Datatypes
        p | send_procs;
        p | send_row_starts;
        p | send_row_indices;
        p | recv_procs;
        p | recv_col_starts;
        p | col_to_proc;

        // Heap Data
        if (p.isUnpacking())
        {
            if (num_sends)
            {
                send_requests = new MPI_Request[num_sends];
                send_buffer = new data_t[size_sends];
            }
            if (num_recvs)
            {
                recv_requests = new MPI_Request[num_recvs];
                recv_buffer = new data_t[size_recvs];
            }
        }
        if (num_sends)
        {
            p(send_requests, num_sends);
            p(send_buffer, size_sends);
        }
        if (num_recvs)
        {
            p(recv_requests, num_recvs);
            p(recv_buffer, size_recvs);
        }
        if (p.isDeleting())
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
        }
    }
#endif

    index_t num_sends;
    index_t num_recvs;
    index_t size_sends;
    index_t size_recvs;
    Array<index_t> send_procs;
    Array<index_t> send_row_starts;
    Array<index_t> send_row_indices;
    Array<index_t> recv_procs;
    Array<index_t> recv_col_starts;
    Array<index_t> col_to_proc;
    MPI_Request* send_requests;
    MPI_Request* recv_requests; 
    data_t* send_buffer;
    data_t* recv_buffer;
};
}
#endif
