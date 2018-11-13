// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "vector.hpp"

using namespace raptor;

/**************************************************************
*****   Vector Set Constant Value
**************************************************************
***** Initializes the vector to a constant value
*****
***** Parameters
***** -------------
***** alpha : data_t
*****    Constant value to set each element of vector to
**************************************************************/
void Vector::set_const_value(data_t alpha)
{
    for (index_t i = 0; i < num_values*b_vecs; i++)
    {
        values[i] = alpha;
    }
}

/**************************************************************
*****   Vector Set Random Values
**************************************************************
***** Initializes each element of the vector to a random
***** value
**************************************************************/
void Vector::set_rand_values()
{
    srand(time(NULL));
    for (index_t i = 0; i < num_values*b_vecs; i++)
    {
        values[i] = ((double)rand()) / RAND_MAX;
    }
}

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
    for (index_t i = 0; i < num_values*b_vecs; i++)
    {
        values[i] += x.values[i]*alpha;
    }
}

/**************************************************************
*****   Vector Copy
**************************************************************
***** Copies each vector value of y into values
*****
***** Parameters
***** -------------
***** y : Vector&
*****    Vector to be copied.  Must have same local rows
*****    and same first row
**************************************************************/
void Vector::copy(const Vector& y)
{
    num_values = y.num_values;
    b_vecs = y.b_vecs;
    values.resize(num_values * b_vecs);
    std::copy(y.values.begin(), y.values.end(), values.begin());
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
**************************************************************/
void Vector::scale(data_t alpha)
{
    for (index_t i = 0; i < num_values; i++)
    {
        values[i] *= alpha;
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
*****   Print Vector
**************************************************************
***** Prints all nonzero elements in vector
*****
***** Parameters
***** -------------
***** vec_name : const char* (optional)
*****    Name to be printed.  Default prints Vec[%d] = %e.
**************************************************************/
void Vector::print(const char* vec_name)
{
    index_t offset;
    printf("Size = %d\n", num_values);
    for (int j = 0; j < b_vecs; j++)
    {
        offset = j * num_values;
        for (int i = 0; i < num_values; i++)
        {
            if (fabs(values[i]) > zero_tol)
                printf("%s[%d][%d] = %e\n", vec_name, j, i, values[i + offset]);
        }
    }
}

/**************************************************************
*****   Vector Element Access
**************************************************************
***** Function overload for element access
*****
***** Returns
***** ------------
***** data_t& element at position passed
**************************************************************/
data_t& Vector::operator[](const int index)
{
    return values[index];
}


data_t Vector::inner_product(Vector& x, data_t* inner_prods)
{
    data_t result = 0.0;

    if (inner_prods == NULL)
    {
        for (int i = 0; i < num_values; i++)
        {
            result += values[i] * x[i];
        }
        return result;
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
                result += values[i + offset] * x[i];
            }
            inner_prods[j] = result;
        }
        return 0;
    }
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

    for (index_t j = 0; j < b_vecs; j++) {
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

    for (index_t j = 0; j < b_vecs; j++)
    {
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            values[i + offset] += y.values[i + offset]*alpha;
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

    for (index_t j = 0; j < b_vecs; j++)
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

    for (index_t j = 0; j < b_vecs; j++)
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
*****   BVector Mult 
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
void BVector::mult(Vector& x, Vector& b)
{
    b.set_const_value(0.0);
    data_t result;
    index_t offset;

    for (index_t j = 0; j < b_vecs; j++)
    {
        result = 0.0;
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            b[i] += values[i + offset] * x[j];
        }
    }
}

/**************************************************************
*****   BVector Mult 
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
    b.set_const_value(0.0);
    data_t result;
    index_t offset;

    for (index_t j = 0; j < b_vecs; j++)
    {
        result = 0.0;
        offset = j * num_values;
        for (index_t i = 0; i < num_values; i++)
        {
            b[i] += values[i + offset] * x[j];
        }
    }
}
