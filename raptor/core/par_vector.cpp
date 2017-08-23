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

std::vector<double>& ParVector::communicate(CommPkg* comm_pkg, MPI_Comm comm)
{
    comm_pkg->communicate(local.data(), comm);
    return comm_pkg->get_recv_buffer();
}

void ParVector::init_comm(CommPkg* comm_pkg, MPI_Comm comm)
{
    comm_pkg->init_comm(local.data(), comm);
}

std::vector<double>& ParVector::complete_comm(CommPkg* comm_pkg)
{
    comm_pkg->complete_comm();
    return comm_pkg->get_recv_buffer();
}

