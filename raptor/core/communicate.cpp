// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "communicate.hpp"

using namespace raptor;

void init_comm_helper(Vector& x, ParComm* comm_pkg, MPI_Comm comm)
{
    CommData* send_data = comm_pkg->send_data;
    CommData* recv_data = comm_pkg->recv_data;

    if (send_data->num_msgs)
    {
        int send_start;
        int send_end;
        int proc;

        data_t* data = x.data();
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
                buffer[j] = data[indices[j]];
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

Vector& complete_comm_helper(ParComm* comm_pkg)
{
    if (comm_pkg->send_data->num_msgs)
    {
        MPI_Waitall(comm_pkg->send_data->num_msgs,
                comm_pkg->send_data->requests,
                MPI_STATUS_IGNORE);
    }

    if (comm_pkg->recv_data->num_msgs)
    {
        MPI_Waitall(comm_pkg->recv_data->num_msgs,
                comm_pkg->recv_data->requests,
                MPI_STATUS_IGNORE);
    }

    return comm_pkg->recv_data->buffer;
}

Vector& communicate_helper(Vector& x, ParComm* comm_pkg, MPI_Comm comm)
{
    init_comm_helper(x, comm_pkg, comm);
    return complete_comm_helper(comm_pkg);
}
