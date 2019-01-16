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
    values.resize(num_values * (b_vecs + P.b_vecs));
    std::copy(P.values.begin(), P.values.end(), values.begin() + num_values * b_vecs);
    b_vecs += P.b_vecs;
}

/**************************************************************
*****   Vector Split 
**************************************************************
***** Splits the vector into t bvecs
*****
***** Parameters 
***** ------------
***** W : Vector&
*****    The Vector to contain the splitting of the vector 
***** t : int
*****    The number of bvecs to split the vector into
***** i : int
*****    The index of the vector in W that should contain
*****    the calling vector's values.
**************************************************************/
void Vector::split(Vector& W, int t, int i)
{
    W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);
    std::copy(values.begin(), values.end(), W.values.begin() + num_values * i);
}

/**************************************************************
*****   Vector Split Range
**************************************************************
***** Splits the vector into t bvecs
***** Splitting the values in the vector across the vectors
***** from block index start to block index stop
*****
***** Parameters 
***** ------------
***** W : Vector&
*****    The Vector to contain the splitting of the vector 
***** t : int
*****    The number of bvecs to split the vector into
***** start : int
*****    The index of the vector in W that should contain
*****    the first portion of the calling vector's values.
**************************************************************/
void Vector::split_range(Vector& W, int t, int start)
{
    W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);

    for (int i = 0; i < num_values; i++)
    {
        W.values[start*num_values + i] = values[i];
        start = (start+1) % t;
    }
}

/**************************************************************
*****   Vector Split Contiguous 
**************************************************************
***** Splits the vector into t bvecs
***** Splitting the values in the vector across the vectors
***** in equal sized contiguous chunks
*****
***** Parameters 
***** ------------
***** W : Vector&
*****    The Vector to contain the splitting of the vector 
***** t : int
*****    The number of bvecs to split the vector into
***** first_global_index : int
*****    The corresponding global index of the first index
*****    in this vector
**************************************************************/
void Vector::split_contig(Vector& W, int t, int first_global_index, int glob_vals)
{
    int glob_index, bvec, pos_in_bvec, end;
    int chunk_size = glob_vals / t;

    W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);

    for (int i = 0; i < num_values; i+= chunk_size)
    {
        glob_index = i + first_global_index;
        bvec = glob_index / t;
        //pos_in_bvec = i % t;
        if (i + chunk_size > num_values) end = num_values;
        else end = chunk_size;
        for (int j = 0; j < chunk_size; j++)
        {
            W.values[bvec*num_values + bvec*chunk_size+j] = values[i+j];
        }
        //pos_in_bvec = i % t;
        //W.values[bvec*num_values + pos_in_bvec] = values[i];
    }
}
