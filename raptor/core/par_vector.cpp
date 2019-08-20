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
*****   ParVector Add ParVector 
**************************************************************
***** Adds a Value to the ParVector 
**************************************************************/
void ParVector::add_val(data_t val, index_t vec, index_t global_n, index_t first_local)
{
    if ((global_n >= first_local) && (global_n < first_local + local_n))
    {
        local->values[vec*local_n + (global_n - first_local)] = val;
    }
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
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    
    if (t == num_procs) local->split(*(W.local), t, rank);
    else local->split_range(*(W.local), t, rank % t);
}

/**************************************************************
*****   ParVector Split Contiguous ParVector 
**************************************************************
***** Splits a ParVector into t bvecs making a ParBVector
***** Storing the local values in equal sized contiguous blocks 
***** for each rank
**************************************************************/
void ParVector::split_contig(ParVector& W, int t, int first_local)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    
    if (t == num_procs) local->split(*(W.local), t, rank);
    else
    {
        int group_size = global_n / t;
        int n;
        for (int i = 0; i < local_n; i++)
        {
            n = first_local + i;
            W.add_val(local->values[i], n/group_size, n, first_local);
        }
    }
}
