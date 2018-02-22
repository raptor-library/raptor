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
    for (index_t i = 0; i < num_values; i++)
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
    for (index_t i = 0; i < num_values; i++)
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
    for (index_t i = 0; i < num_values; i++)
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
    values.resize(num_values);
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
data_t Vector::norm(index_t p)
{
    data_t result = 0.0;
    double val;
    for (index_t i = 0; i < num_values; i++)
    {
        val = values[i];
        if (fabs(val) > zero_tol)
            result += pow(val, p);
    }
    return pow(result, 1.0/p);
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
    printf("Size = %d\n", num_values);
    for (int i = 0; i < num_values; i++)
    {
        if (fabs(values[i]) > zero_tol)
            printf("%s[%d] = %e\n", vec_name, i, values[i]);
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


data_t Vector::inner_product(Vector& x)
{
    data_t result = 0.0;

    for (int i = 0; i < num_values; i++)
    {
        result += values[i] * x[i];
    }

    return result;
}


