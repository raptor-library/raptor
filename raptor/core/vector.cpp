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

/**************************************************************
*****   Vector Append
**************************************************************
***** Appends P to the Vector by adding P as additional
***** vectors in the Vector and increases the block size
*****
***** Parameters 
***** ------------
***** P : Vector&
*****    The Vector to append
**************************************************************/
void Vector::append(Vector& P)
{
    index_t old_num_values = num_values;    
    b_vecs += P.b_vecs;
    resize(num_values + P.num_values);
    values.resize(num_values * b_vecs);
    std::copy(P.values.begin(), P.values.end(), values.begin() + old_num_values);
}
