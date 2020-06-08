// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "vector_cuda.hpp"
#include "vector_cuda_kernels.cuh"

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
    printf("before cuda kernel\n");
    set_const_value_kernel<<<tblocks,blocksize>>>(alpha, dev_ptr, values.size());
    printf("after cuda kernel\n");
}

/**************************************************************
*****   Vector Set Random Values
**************************************************************
***** Initializes each element of the vector to a random
***** value
**************************************************************/
void Vector::set_rand_values(int seed)
{
    set_rand_values_kernel<<<tblocks,blocksize>>>(seed, dev_ptr, values.size());
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

    // Allocate on device memory and copy 
    cudaMalloc(&dev_ptr, num_values * b_vecs * sizeof(double));
    copy_kernel<<<tblocks,blocksize>>>(y.dev_ptr, dev_ptr, num_values * b_vecs);
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
    // Leaving alone -- assuming user will copy from device
    // first then print
    index_t offset;
    printf("Size = %d\n", num_values);
    for (int j = 0; j < b_vecs; j++)
    {
        offset = j * num_values;
        for (int i = 0; i < num_values; i++)
        {
            if (fabs(values[i + offset]) > zero_tol)
                printf("%s[%d] = %e\n", vec_name, j, i, values[i + offset]);
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
    // Leaving alone -- assuming user will copy from device
    // first then print
    return values[index];
}

/**************************************************************
*****   Vector Copy from Device 
**************************************************************
***** Function that copies vector values from device into
***** host vector array
**************************************************************/
void Vector::copy_from_device()
{
    cudaMemcpy(dev_ptr, values.data(), values.size()*sizeof(double), cudaMemcpyHostToDevice);
}

/**************************************************************
*****   Vector Copy from Host 
**************************************************************
***** Function that copies vector values from host into
***** device vector array
**************************************************************/
void Vector::copy_from_host()
{
    cudaMemcpy(values.data(), dev_ptr, values.size()*sizeof(double), cudaMemcpyDeviceToHost);
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
    // RESIZE ON DEVICE VECTOR
    // STORE NEW VALUES IN APPROPRIATE PLACE ON DEVICE
    values.resize(num_values * (b_vecs + P.b_vecs));
    std::copy(P.values.begin(), P.values.end(), values.begin() + (num_values * b_vecs));
    b_vecs += P.b_vecs;
}

/**************************************************************
*****   Vector Split 
**************************************************************
***** Splits the vector into t b_vecs
*****
***** Parameters 
***** ------------
***** W : Vector&
*****    The Vector to contain the resulting split Vector
***** t : int
*****    The number of b_vecs to split the Vector into
***** i : int
*****    The index of the Vector in W that should contain the
*****    the calling Vector's values.
**************************************************************/
void Vector::split(Vector& W, int t, int i)
{
    // COPY VECTOR FROM DEV 
    // SPLIT VECTOR ON HOST	
    W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);
    std::copy(values.begin(), values.end(), W.values.begin() + (num_values * i));
    // RESIZE VECTOR ON DEV
    // COPY VECTOR BACK TO DEV
}

/**************************************************************
*****   Vector Split Range 
**************************************************************
***** Splits the vector into t b_vecs
***** Splitting the values in the vector across the vectors
***** from block index start to block index stop 
*****
***** Parameters 
***** ------------
***** W : Vector&
*****    The Vector to contain the resulting split Vector
***** t : int
*****    The number of b_vecs to split the Vector into
***** start : int
*****    The index of the Vector in W that should contain the
*****    first portion of the calling Vector's values.
**************************************************************/
void Vector::split_range(Vector& W, int t, int start)
{
    // COPY VECTOR FROM DEV 
    // SPLIT VECTOR ON HOST	
    W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);

    for (int i = 0; i < num_values; i++)
    {
        W.values[start*num_values + i] = values[i];
        start = (start + 1) % t;
    }
    // RESIZE VECTOR ON DEV
    // COPY VECTOR BACK TO DEV
}

/**************************************************************
*****   Vector Split Contiguous 
**************************************************************
***** Splits the vector into t b_vecs
***** Splitting the values in the vector across the vectors
***** in equal sized contiguous chunks 
*****
***** Parameters 
***** ------------
***** W : Vector&
*****    The Vector to contain the resulting split Vector
***** t : int
*****    The number of b_vecs to split the Vector into
***** first_global_index : int
*****    The corresponding global index of the first index
*****    in this vector 
**************************************************************/
void Vector::split_contig(Vector& W, int t, int first_global_index, int glob_vals)
{
    // COPY VECTOR FROM DEV 
    // SPLIT VECTOR ON HOST	
    int glob_index, bvec, pos_in_bvec, end;
    int chunk_size = glob_vals / t;

    W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);

    for (int i = 0; i < num_values; i+= chunk_size)
    {
        glob_index = i + first_global_index;
        bvec = glob_index / t;
        if (i + chunk_size > num_values) end = num_values;
        else end = chunk_size;
        for (int j = 0; j < chunk_size; j++)
        {
            W.values[bvec*num_values + bvec*chunk_size + j] = values[i + j];
        }
    }
    
    // RESIZE VECTOR ON DEV
    // COPY VECTOR BACK TO DEV
}

