// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/vector.hpp"

using namespace raptor;

/**************************************************************
*****   Vector AXPY
**************************************************************
***** Multiplies the vector x by a constant, alpha, and then
***** sums each element with corresponding local entry 
*****
***** Parameters
***** -------------
***** x : Vector&
*****    Vector to be summed with
***** alpha : data_t
*****    Constant value to multiply each element of vector by
**************************************************************/
void Vector::axpy(Vector& x, data_t alpha)
{
    index_t i;
    /*#pragma omp parallel for default(none) private(i) \
        shared(num_values, values, x, b_vecs, alpha) schedule(static)*/
    for (i = 0; i < num_values*b_vecs; i++)
    {
        values[i] += x.values[i]*alpha;
    }
}

/**************************************************************
*****   Vector AXPY_ij
**************************************************************
***** Multiplies vector i in the bvector by a constant, 
***** alpha, and then sums each element with corresponding entry 
***** of column j in y
*****
***** Parameters
***** -------------
***** y : Vector& y
*****    Vector to be summed with
***** i : index_t
*****    Column of local vector for axpy
***** j : index_t
*****    Column of y for axpy
***** alpha : data_t
*****    Constant value to multiply each element of column by
**************************************************************/
void Vector::axpy_ij(Vector& y, index_t i, index_t j, data_t alpha)
{
    index_t k;
    /*#pragma omp parallel for default(none) private(k) \
        shared(num_values, values, y, i, j, alpha) schedule(static)*/
    for (k = 0; k < num_values; k++)
    {
        values[i*num_values + k] += y.values[j*num_values + k] * alpha;
    }
}

/**************************************************************
*****   Vector Scale
**************************************************************
***** Multiplies each element of the vector by a constant value
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Constant value to set multiply element of vector by
***** alphas : data_t*
*****    Constant values to multiply element of each vector
*****    in block vector by  
**************************************************************/
void Vector::scale(data_t alpha, data_t* alphas)
{
    if (alphas == NULL)
    {
        index_t i;
        /*#pragma omp parallel for default(none) private(i) \
            shared(num_values, values, alpha) schedule(static)*/
        for (index_t i = 0; i < num_values; i++)
        {
            values[i] *= alpha;
        }
    }
    else
    {
        index_t offset;
        index_t j;
        /*#pragma omp parallel for default(none) private(j, offset) \
            shared(num_values, values, alphas, b_vecs) schedule(static)*/
        for (j = 0; j < b_vecs; j++)
        {
            offset = j * num_values;
            for (index_t i = 0; i < num_values; i++)
            {
                values[i + offset] *= alphas[j];
            }
        }
    }
}

/**************************************************************
*****   Vector Norm
**************************************************************
***** Calculates the P norm of the vector (for a given P)
*****
***** Parameters
***** -------------
***** p : index_t
*****    Determines which p-norm to calculate
**************************************************************/
data_t Vector::norm(index_t p, data_t* norms)
{
    data_t result = 0.0;
    double val;

    if (norms == NULL)
    {    
        for (index_t i = 0; i < num_values; i++)
        {
            val = values[i];
            if (fabs(val) > zero_tol)
                result += pow(val, p);
        }
        return pow(result, 1.0/p);
    }
    else
    {
        index_t offset;
        for (index_t j = 0; j < b_vecs; j++)
        {
            result = 0.0;
            offset = j * num_values;
            for (index_t i = 0; i < num_values; i++)
            {
                val = values[i + offset];
                if (fabs(val) > zero_tol)
                    result += pow(val, p);
            }
            norms[j] = pow(result, 1.0/p);
        }
        return 0;
    }
}

/**************************************************************
*****   Vector Inner Product
**************************************************************
***** Calculates the inner product of the given vector with 
***** input vector
*****
***** Parameters
***** -------------
***** ADD 
**************************************************************/
data_t Vector::inner_product(Vector& x, data_t* inner_prods)
{
    data_t result = 0.0;

    if (inner_prods == NULL)
    {
        index_t i;
        /*#pragma omp parallel for default(none) private(i) shared(num_values, values, x) \
            reduction(+:result) schedule(static)*/
        for (i = 0; i < num_values; i++)
        {
            result += values[i] * x[i];
        }
        return result;
    }
    else
    {
        index_t offset;
        if (x.b_vecs == 1)
        {
            index_t j;
            /*#pragma omp parallel for default(none) private(j, result, offset) \
                shared(num_values, values, x, inner_prods) schedule(static)*/
            for (j = 0; j < b_vecs; j++)
            {
                result = 0.0;
                offset = j * num_values;
                for (index_t i = 0; i < num_values; i++)
                {
                    result += values[i + offset] * x[i];
                }
                inner_prods[j] = result;
            }
        }
        else
        {
            index_t j;
            /*#pragma omp parallel for default(none) private(j, result, offset) \
                shared(num_values, values, x, inner_prods) schedule(static)*/
            for (j = 0; j < b_vecs; j++)
            {
                result = 0.0;
                offset = j * num_values;
                for (index_t i = 0; i < num_values; i++)
                {
                    result += values[i + offset] * x[i + offset];
                }
                inner_prods[j] = result;
            }
        }
        return 0;
    }
}

/**************************************************************
*****   Vector Inner Product 
**************************************************************
***** Calculates the inner product of the ith column of the
***** Vector with the jth column of x 
*****
***** Parameters
***** -------------
***** x : Vector&
*****   Vector with which to perform inner product
***** i : index_t
*****   Column of calling Vector for inner product
***** j : index_t
*****   Column of x for inner product
**************************************************************/
data_t Vector::inner_product(Vector& x, index_t i, index_t j)
{
    data_t result = 0.0;
    
    index_t k;
    /*#pragma omp parallel for default(none) private(k) shared(num_values, values, x, i, j) \
        reduction(+:result) schedule(static)*/
    for (k = 0; k < num_values; k++)
    {
        result += values[i*num_values + k] * x.values[j*num_values + k];
    }
    return result;
}

/**************************************************************
*****   BVector AXPY
**************************************************************
***** Multiplies the vector x by a constant, alpha, and then
***** sums each element with corresponding local entry for
***** each vector in the BVector 
*****
***** Parameters
***** -------------
***** x : Vector&
*****    Vector to be summed with
***** alpha : data_t
*****    Constant value to multiply each element of vector by
**************************************************************/
void BVector::axpy(Vector& x, data_t alpha)
{
    index_t offset;
    index_t j; 

    /*#pragma omp parallel for default(none) private(j, offset) \
        shared(num_values, values, x, b_vecs, alpha) schedule(static)*/
    for (j = 0; j < b_vecs; j++) {
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            values[i + offset] += x.values[i]*alpha;
        }
    }
}

/**************************************************************
*****   BVector AXPY
**************************************************************
***** Multiplies each vector in y by a constant, alpha, and then
***** sums each entry in each vector with corresponding local entry
***** and vector in the BVector 
*****
***** Parameters
***** -------------
***** y : BVector&
*****    BVector to be summed with
***** alpha : data_t
*****    Constant value to multiply each element of each vector by
**************************************************************/
void BVector::axpy(BVector& y, data_t alpha)
{
    index_t offset;

    index_t j;
    /*#pragma omp parallel for default(none) private(j, offset) \
        shared(num_values, values, y, alpha, b_vecs) schedule(static)*/
    for (j = 0; j < b_vecs; j++)
    {
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            values[i + offset] += y.values[i + offset]*alpha;
        }
    }
}

/**************************************************************
*****   BVector Scale
**************************************************************
***** Multiplies each element of the vector by a constant value
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Constant value to set multiply element of vector by
***** alphas : data_t*
*****    Constant values to multiply element of each vector
*****    in block vector by  
**************************************************************/
void BVector::scale(data_t alpha, data_t* alphas)
{
    if (alphas == NULL)
    {
        index_t i;
        /*#pragma omp parallel for default(none) private(i) \
            shared(num_values, values, alpha) schedule(static)*/
        for (i = 0; i < num_values; i++)
        {
            values[i] *= alpha;
        }
    }
    else
    {
        index_t offset;
        index_t j;
        /*#pragma omp parallel for default(none) private(j, offset) \
            shared(num_values, values, alphas, b_vecs) schedule(static)*/
        for (j = 0; j < b_vecs; j++)
        {
            offset = j * num_values;
            for (index_t i = 0; i < num_values; i++)
            {
                values[i + offset] *= alphas[j];
            }
        }
    }
}

/**************************************************************
*****   BVector Norm
**************************************************************
***** Calculates the P norm of each vector (for a given P)
***** in the block vector
*****
***** Parameters
***** -------------
***** p : index_t
*****    Determines which p-norm to calculate
**************************************************************/
data_t BVector::norm(index_t p, data_t* norms)
{
    data_t result;
    index_t offset;
    double val;

    for (index_t j = 0; j < b_vecs; j++)
    {
        result = 0.0;
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            val = values[i + offset];
            if (fabs(val) > zero_tol)
                result += pow(val, p);
        }
        norms[j] = pow(result, 1.0/p);
    }

    return 0;
}

/**************************************************************
*****   BVector Inner Product 
**************************************************************
***** Calculates the inner product of every vector in the
***** block vector with the vector x
*****
***** Parameters
***** -------------
***** x : Vector&
*****    Vector to calculate inner product with
**************************************************************/
data_t BVector::inner_product(Vector& x, data_t* inner_prods)
{
    data_t result;
    index_t offset;

    index_t j;
    /*#pragma omp parallel for default(none) private(j, result, offset) \
        shared(num_values, values, x, inner_prods) schedule(static)*/
    for (j = 0; j < b_vecs; j++)
    {
        result = 0.0;
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            result += values[i + offset] * x[i];
        }
        inner_prods[j] = result;
    }

    return 0;
}

/**************************************************************
*****   BVector Inner Product 
**************************************************************
***** Calculates the inner product of every vector in the
***** block vector with each corresponding vector in y
*****
***** Parameters
***** -------------
***** y : BVector&
*****    BVector to calculate inner product with
**************************************************************/
data_t BVector::inner_product(BVector& y, data_t* inner_prods)
{
    data_t result;
    index_t offset;

    index_t j;
    /*#pragma omp parallel for default(none) private(j, result, offset) \
        shared(num_values, values, y, inner_prods) schedule(static)*/
    for (j = 0; j < b_vecs; j++)
    {
        result = 0.0;
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            result += values[i + offset] * y[i + offset];
        }
        inner_prods[j] = result;
    }

    return 0;
}

/**************************************************************
*****   BVector Mult_T 
**************************************************************
***** Multiplies the transpose of BVector by the BVector X - dense 
***** matrix multiplication
*****
***** Parameters
***** -------------
***** X : BVector&
*****    BVector to multiply with
***** B : BVector&
*****    BVector to hold result
**************************************************************/
void Vector::mult_T(Vector& X, Vector& B)
{
    // Resize B
    B.b_vecs = X.b_vecs;
    B.resize(b_vecs);

    data_t result;
    index_t x;
    /*#pragma omp parallel for default(none) private(x, result) \
        shared(X, b_vecs, num_values, values, B) schedule(static)*/
    for (x = 0; x < X.b_vecs; x++)
    {
        for (index_t v = 0; v < b_vecs; v++)
        {
            result = 0.0;
            for (index_t i = 0; i < num_values; i++)
            {
                result += values[v*num_values + i] * X[x*X.num_values + i];
            }
            B.values[x*B.num_values + v] = result;
        }
    }
}

/**************************************************************
*****   Vector Mult 
**************************************************************
***** Multiplies the BVector by the Vector x as if the 
***** the BVector were a dense matrix
*****
***** Parameters
***** -------------
***** x : Vector&
*****    Vector to multiply with
***** b : Vector&
*****    Vector to hold result
**************************************************************/
void Vector::mult(Vector& x, Vector& b)
{
    // Resize B
    b.b_vecs = x.b_vecs;
    b.resize(num_values);
    
    data_t result;
    index_t j;
    /*#pragma omp parallel for default(none) private(j, result) \
        shared(num_values, values, x, b) schedule(static)*/
    for (j = 0; j < x.b_vecs; j++)
    {
        for (index_t v = 0; v < num_values; v++)
        {
            result = 0.0;
            for (index_t i = 0; i < b_vecs; i++)
            {
                result += values[i*num_values + v] * x.values[j*b_vecs + i];
            }
            b.values[j*num_values + v] = result;
        }
    }
}