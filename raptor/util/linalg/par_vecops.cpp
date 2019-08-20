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
*****   ParVector Norm
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
        result = local->norm(p);
        result = pow(result, p); // undoing root of p from local operation
    }
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, &result, 1, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    return pow(result, 1./p);
}


/**************************************************************
*****   ParVector Inner Product 
**************************************************************
***** Calculates the inner product between 2 global vectors
*****
***** Parameters
***** -------------
***** x : ParVector&
*****   Global vector with which to perform inner product 
**************************************************************/
data_t ParVector::inner_product(ParVector& x)
{
    data_t inner_prod = 0.0;

    if (local_n != x.local_n)
    {
        int rank;
        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }

    if (local_n)
    {
        inner_prod = local->inner_product(*(x.local));
    }

    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, &inner_prod, 1, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    
    return inner_prod;
}

/**************************************************************
*****   ParBVector AXPY IJ 
**************************************************************
***** Multiplies the vector i in the local bvector by a constant
***** alpha and then sums each element with corresponding entry
***** of column j in y's local bvector 
*****
***** Parameters
***** -------------
***** y : ParBVector&
*****   Global vector with which to perform axpy 
***** i : index_t
*****   Column of local bvector for axpy
***** j : index_t
*****   Column of y's local bvector for axpy
***** alpha : data_t
*****   Constant value to multiply each element of column by 
**************************************************************/
void ParBVector::axpy_ij(ParBVector& y, index_t i, index_t j, data_t alpha)
{
    if (local_n) local->axpy_ij(*(y.local), i, j, alpha);
}

/**************************************************************
*****   ParBVector Scale
**************************************************************
***** Multiplies the local vector by a constant, alpha
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Constant value to multiply each element of vector by
**************************************************************/
void ParBVector::scale(data_t alpha, data_t* alphas)
{
    //if (local_n) local->scale(alpha, alphas);
    if (local_n) local->scale(alpha);
}

/**************************************************************
*****   ParBVector Norm 
**************************************************************
***** Calculates the p norm of each global vector in the
***** ParBVector (for a given p) 
*****
***** Parameters
***** -------------
***** X : ParVector*
*****   ParVector of which to calculate p-norm
***** p : index_t
*****   Determines whcih p-norm to calculate 
***** norms : data_t*
*****   Array to hold norm of each vector
**************************************************************/
void block_norm_helper(ParVector* X, index_t p, data_t* norms)
{
    data_t temp;
    if (X->local_n)
    {
        temp = X->local->norm(p, norms);
        for (int i = 0; i < X->local->b_vecs; i++)
        {
            norms[i] = pow(norms[i], p); // undoing root of p from local operation
        }
    }

    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, norms, X->local->b_vecs, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);

    for (int i = 0; i < X->local->b_vecs; i++)
    {
        norms[i] = pow(norms[i], 1./p);
    }
}

/**************************************************************
*****   ParBVector Norm 
**************************************************************
***** Calculates the p norm of each global vector in the
***** ParBVector (for a given p) 
*****
***** Parameters
***** -------------
***** p : index_t
*****   Determines whcih p-norm to calculate 
**************************************************************/
data_t ParBVector::norm(index_t p, data_t* norms)
{
    if (local->b_vecs == 1) return ParVector::norm(p);
    else block_norm_helper(this, p, norms);

    return 0;
}

/**************************************************************
*****   ParBVector Inner Product 
**************************************************************
***** Calculates the inner product of every vector in the
***** parbvector with every vector in x 
*****
***** Parameters
***** -------------
***** x : ParBVector&
*****   Global vector with which to calculate inner products 
***** inner_prods : data_t*
*****   Inner products of every corresponding vector in each
*****   ParBVector
**************************************************************/
data_t ParBVector::inner_product(ParBVector& x, data_t* inner_prods)
{
    data_t temp;
    data_t inner_prod = 0.0;

    if (local_n != x.local_n)
    {
        int rank;
        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }

    if (local_n)
    {
        temp = local->inner_product(*(x.local), inner_prods);
    }

    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, inner_prods, local->b_vecs, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    
    return 0;
}

/**************************************************************
*****   ParBVector Mult_T 
**************************************************************
***** Calculates the transpose multiplication of the ParBVector
***** with the ParVector x - spmv with dense matrix 
*****
***** Parameters
***** -------------
***** x : ParVector& 
*****   Perform transpose multiplication with this ParVector
***** b : Vector&
*****   Vector in which to store result 
**************************************************************/
void ParBVector::mult_T(ParVector& x, Vector& b)
{
    data_t temp;
    if (local_n)
    {
        local->mult_T(*(x.local), b);
    }
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, &(b[0]), local->b_vecs * x.local->b_vecs, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
}

/**************************************************************
*****   ParBVector Mult 
**************************************************************
***** Calculates the multiplication of the ParBVector
***** with the Vector x on the local block 
*****
***** Parameters
***** -------------
***** x : Vector& 
*****   Perform multiplication with this Vector on local block
***** b : ParVector&
*****   Store result in local portion of b 
**************************************************************/
void ParBVector::mult(Vector& x, ParVector& b)
{
    b.resize(global_n, local_n);
    local->mult(x, *(b.local));
}

/**************************************************************
*****   ParBVector Inner Product IJ 
**************************************************************
***** Calculates the inner product of the ith column of the 
***** ParBVector with the jth column of x  
*****
***** Parameters
***** -------------
***** x : ParBVector&
*****   Global vector with which to calculate inner product 
***** i : index_t
*****   Column of calling ParBVector for inner product 
***** j : index_t
*****   Column of x for inner product
**************************************************************/
data_t ParBVector::inner_product(ParBVector& x, index_t i, index_t j)
{
    data_t temp;
    
    if (local_n != x.local_n)
    {
        int rank;
        RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }

    if (local_n)
    {
        temp = local->inner_product(*(x.local), i, j);
    }

    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, &temp, 1, RAPtor_MPI_DATA_T, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    
    return temp;
}
