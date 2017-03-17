// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_vector.hpp"

using namespace raptor;


/**************************************************************
*****   Vector AXPY
**************************************************************
***** Multiplies the local vector by a constant, alpha, and then
***** sums each element with corresponding entry of Y
*****
***** Parameters
***** -------------
***** y : ParVector* y
*****    Vector to be summed with
***** alpha : data_t
*****    Constant value to multiply each element of vector by
**************************************************************/
void ParVector::axpy(ParVector* x, data_t alpha)
{
    if (local_n)
    {
        local.axpy(x->local, alpha);
    }
}

/**************************************************************
*****   Vector Copy
**************************************************************
***** Copies values of local vector in y into local 
*****
***** Parameters
***** -------------
***** y : ParVector* y
*****    ParVector to be copied
**************************************************************/
void ParVector::copy(ParVector* y)
{
    if (local_n)
        local.copy(y->local);
}

/**************************************************************
*****   Vector Scale
**************************************************************
***** Multiplies the local vector by a constant, alpha
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Constant value to multiply each element of vector by
**************************************************************/
void ParVector::scale(data_t alpha)
{
    if (local_n)
    {
        local.scale(alpha);
    }
}

/**************************************************************
*****   ParVector Set Constant Value
**************************************************************
***** Sets each element of the local vector to a constant value
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Value to set each element of local vector to
**************************************************************/
void ParVector::set_const_value(data_t alpha)
{
    if (local_n)
    {
        local.set_const_value(alpha);
    }
}

/**************************************************************
*****   ParVector Set Random Values
**************************************************************
***** Sets each element of the local vector to a random value
**************************************************************/
void ParVector::set_rand_values()
{
    if (local_n)
    {
        local.set_rand_values();
    }
}

/**************************************************************
*****   Vector Norm
**************************************************************
***** Calculates the P norm of the global vector (for a given P)
*****
***** Parameters
***** -------------
***** p : index_t
*****    Determines which p-norm to calculate
**************************************************************/
data_t ParVector::norm(index_t p)
{
    data_t result = 0.0;
    if (local_n)
    {
        result = local.norm(p);
        result = pow(result, p); // undoing root of p from local operation
    }
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    return pow(result, 1./p);
}

Vector& ParVector::communicate(ParComm* comm_pkg, MPI_Comm comm)
{
    init_comm(comm_pkg, comm);
    return complete_comm(comm_pkg);
}

void ParVector::init_comm(ParComm* comm_pkg, MPI_Comm comm)
{
    CommData* send_data = comm_pkg->send_data;
    CommData* recv_data = comm_pkg->recv_data;

    if (send_data->num_msgs)
    {
        int send_start;
        int send_end;
        int proc;

        data_t* local_data = local.data();
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
                buffer[j] = local_data[indices[j]];
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

Vector& ParVector::complete_comm(ParComm* comm_pkg)
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

