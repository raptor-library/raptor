// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_vector.hpp"

using namespace raptor;


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
        local->set_const_value(alpha);
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
        local->set_rand_values();
    }
}

/**************************************************************
*****   ParVector Append ParVector
**************************************************************
***** Appends a ParVector to the ParVector
**************************************************************/
void ParBVector::append(ParBVector& P)
{
    local->append(*(P.local));
}

/**************************************************************
*****   ParVector Split ParVector
**************************************************************
***** Splits a ParVector into t bvecs making a ParBVector
***** Storing the local values in W at the appropriate bvec
***** index
**************************************************************/
void ParVector::split(ParVector& W, int t)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (t == num_procs) local->split(*(W.local), t, rank);
    else local->split_range(*(W.local), t, rank % t);
    //else if (t < num_procs) local->split(*(W.local), t, rank % t);
}
