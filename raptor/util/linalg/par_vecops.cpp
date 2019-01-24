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

data_t ParVector::inner_product_timed(ParVector& x, aligned_vector<double>& times, data_t* inner_prods)
{
    double start, stop;
    data_t inner_prod;

    if (local_n != x.local_n)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }

    start = MPI_Wtime();
    if (local_n)
    {
        inner_prod = local->inner_product(*(x.local));
    }
    stop = MPI_Wtime();
    times[2] += (stop - start);

    start = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, &inner_prod, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    stop = MPI_Wtime();
    times[0] += (stop - start);
    
    return inner_prod;
}

/**************************************************************
*****   ParBVector AXPY
**************************************************************
***** Multiplies vector i in the local bvector by a constant, 
***** alpha, and then sums each element with corresponding entry 
***** of column j in y's local bvector
*****
***** Parameters
***** -------------
***** y : ParBVector* y
*****    ParBVector to be summed with
***** i : index_t
*****    Column of local bvector for axpy
***** j : index_t
*****    Column of y's local bvector for axpy
***** alpha : data_t
*****    Constant value to multiply each element of column by
**************************************************************/
void ParBVector::axpy_ij(ParBVector& y, index_t i, index_t j, data_t alpha)
{
    if (local_n)
    {
        local->axpy_ij(*(y.local), i, j, alpha);
    }
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

/**************************************************************
*****   ParBVector Inner Product 
**************************************************************
***** Calculates the inner product of every vector in the
***** parbvector with every vector in x 
*****
***** Parameters
***** -------------
***** x : ParBVector&
*****   ParBVector with which to calculate inner products
***** inner_prods : data_t*
*****   Inner products of every corresponding vector in each
*****   ParBVector
**************************************************************/
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
    MPI_Allreduce(MPI_IN_PLACE, &(b[0]), local->b_vecs * x.local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
}

void ParBVector::mult_T_timed(ParVector& x, Vector& b, aligned_vector<double>& times)
{
    double start, stop;

    start = MPI_Wtime();
    data_t temp;
    if (local_n)
    {
        local->mult_T(*(x.local), b);
    }
    stop = MPI_Wtime();
    times[2] += (stop - start);

    start = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, &(b[0]), local->b_vecs * x.local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    stop = MPI_Wtime();
    times[0] += (stop - start);
}

/**************************************************************
*****   ParBVector Mult_T 
**************************************************************
***** Calculates the transpose multiplication of the ParBVector
***** with the ParBVector x - dense matmult_T
*****
***** Parameters
***** -------------
***** x : ParBVector&
*****   Perform transpose multiplication with this ParBVector
***** b : Vector&
*****   Vector in which to store result
**************************************************************/
void ParBVector::mult_T(ParBVector& x, BVector& b)
{
    if (local_n)
    {
        local->mult_T(*(x.local), b);
    }
    MPI_Allreduce(MPI_IN_PLACE, &(b[0]), local->b_vecs*x.local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
}

void ParBVector::mult_T_timed(ParBVector& x, BVector& b, aligned_vector<double>& times)
{
    double start, stop;

    start = MPI_Wtime();
    if (local_n)
    {
        local->mult_T(*(x.local), b);
    }
    stop = MPI_Wtime();
    times[2] += (stop - start);

    start = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, &(b[0]), local->b_vecs*x.local->b_vecs, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
    stop = MPI_Wtime();
    times[0] += (stop - start);
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
    b.resize(global_n, local_n, first_local);
    local->mult(x, *(b.local));
}

/**************************************************************
*****   ParBVector Inner Product 
**************************************************************
***** Calculates the inner product of the ith column of the
***** ParBVector with the jth column of x 
*****
***** Parameters
***** -------------
***** x : ParBVector&
*****   ParBVector with which to perform inner product
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
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("Error.  Cannot perform inner product.  Dimensions do not match.\n");
        exit(-1);
    }
    if (local_n)
    {
        temp = local->inner_product(*(x.local), i, j);
    }
    MPI_Allreduce(MPI_IN_PLACE, &temp, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);
   
    return temp;
}
