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

Vector& ParVector::communicate(TAPComm* comm_pkg, MPI_Comm comm)
{
    int idx;

    // Initial redistribution among node
    init_comm_helper(local, comm_pkg->local_S_par_comm, comm_pkg->local_comm);
    Vector& S_recv = complete_comm_helper(comm_pkg->local_S_par_comm);

    // Inter-node communication
    init_comm_helper(S_recv, comm_pkg->global_par_comm, MPI_COMM_WORLD);
    Vector& G_recv = complete_comm_helper(comm_pkg->global_par_comm);

    // Redistributing recvd inter-node values
    init_comm_helper(G_recv, comm_pkg->local_R_par_comm, comm_pkg->local_comm);
    Vector& R_recv = complete_comm_helper(comm_pkg->local_R_par_comm);

    // Messages with origin and final destination on node
    init_comm_helper(local, comm_pkg->local_L_par_comm, comm_pkg->local_comm);
    Vector& L_recv = complete_comm_helper(comm_pkg->local_L_par_comm);

    // Add values from L_recv and R_recv to appropriate positions in 
    // Vector recv
    for (int i = 0; i < R_recv.size; i++)
    {
        idx = comm_pkg->R_to_orig[i];
        comm_pkg->recv_buffer.values[idx] = R_recv.values[i];
    }

    for (int i = 0; i < L_recv.size; i++)
    {
        idx = comm_pkg->L_to_orig[i];
        comm_pkg->recv_buffer.values[idx] = L_recv.values[i];
    }

    return comm_pkg->recv_buffer;
}

Vector& ParVector::communicate(ParComm* comm_pkg, MPI_Comm comm)
{
    return communicate_helper(local, comm_pkg, comm);
}

void ParVector::init_comm(ParComm* comm_pkg, MPI_Comm comm)
{
    init_comm_helper(local, comm_pkg, comm);
}

Vector& ParVector::complete_comm(ParComm* comm_pkg)
{
    return complete_comm_helper(comm_pkg);
}

