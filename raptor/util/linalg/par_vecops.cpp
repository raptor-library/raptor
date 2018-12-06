// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/par_vector.hpp"

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
void ParVector::axpy(ParVector& x, data_t alpha)
{
    if (local_n)
    {
        local->axpy(*(x.local), alpha);
    }
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
        local->scale(alpha);
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
data_t ParVector::norm(index_t p, data_t* norms)
{
    data_t result;
    if (local_n)
    {
        result = local->norm(p);
        result = pow(result, p); // undoing root of p from local operation
    }
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    return result;
}


data_t ParVector::inner_product(ParVector& x, data_t* inner_prods)
{
    data_t inner_prod;

    if (local_n != x.local_n)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }

    if (local_n)
    {
        inner_prod = local->inner_product(*(x.local));
    }

    MPI_Allreduce(MPI_IN_PLACE, &inner_prod, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    
    return inner_prod;
}

/**************************************************************
*****   ParBVector AXPY
**************************************************************
***** Multiplies each vector in the local bvector by a constant, 
***** alpha, and then sums each element with corresponding entry 
***** of y
*****
***** Parameters
***** -------------
***** y : ParBVector* y
*****    ParBVector to be summed with
***** alpha : data_t
*****    Constant value to multiply each element of bvector by
**************************************************************/
/*void ParBVector::axpy(ParBVector& y, data_t alpha)
{
    if (local_n)
    {
        local->axpy(*(y.local), alpha);
    }
}*/

/**************************************************************
*****   ParBVector Scale
**************************************************************
***** Multiplies the local vector by a constant, alpha
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Constant value to multiply each element of vector by
***** alphas : data_t*
*****    Constant values to multiply element of each vector
*****    in block vector by  
**************************************************************/
void ParBVector::scale(data_t alpha, data_t* alphas)
{
    if (local_n)
    {
        local->scale(alpha, alphas);
    }
}

/**************************************************************
*****   ParBVector Norm
**************************************************************
***** Calculates the P norm of each global vector in the 
***** ParBVector(for a given P)
*****
***** Parameters
***** -------------
***** p : index_t
*****    Determines which p-norm to calculate
**************************************************************/
data_t ParBVector::norm(index_t p, data_t* norms)
{
    data_t temp;
    if (local_n)
    {
        temp = local->norm(p, norms);
        for (int i = 0; i < local->b_vecs; i++)
        {
            norms[i] = pow(norms[i], p); // undoing root of p from local operation
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, norms, local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < local->b_vecs; i++)
    {
        norms[i] = pow(norms[i], 1./p);
    }

    return 0;
}

data_t ParBVector::inner_product(ParBVector& x, data_t* inner_prods)
{
    data_t temp;
    if (local_n != x.local_n)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }

    if (local_n)
    {
        temp = local->inner_product(*(x.local), inner_prods);
    }
    MPI_Allreduce(MPI_IN_PLACE, inner_prods, local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    
    return 0;
}

void ParBVector::mult_T(ParVector& x, Vector& b)
{
    data_t temp;
    if (local_n)
    {
        //temp = local->inner_product(*(x.local), b);
        local->mult_T(*(x.local), b);
    }
    MPI_Allreduce(MPI_IN_PLACE, &(b[0]), local->b_vecs * x.local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
}

void ParBVector::mult_T(ParBVector& x, BVector& b)
{
    if (local_n)
    {
        local->mult_T(*(x.local), b);
    }
    MPI_Allreduce(MPI_IN_PLACE, &(b[0]), local->b_vecs*x.local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
}

void ParBVector::mult(Vector& x, ParVector& b)
{
    b.resize(global_n, local_n, first_local);
    local->mult(x, *(b.local));
}
