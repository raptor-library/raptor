// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_VECTOR_CUDA_KERNELS_CUH_
#define RAPTOR_CORE_VECTOR_CUDA_KERNELS_CUH_

// REMOVE ALL MENTIONS OF VECTOR -- GOING TO PASS CUDA KERNELS DATA POINTERS
// DOUBLES, ETC...., NOTHING RELATED RAPTOR VECTOR CLASS

#include <curand.h>
#include <curand_kernel.h>

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
__global__ void set_const_value_kernel(data_t alpha, data_t* vec, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i=tid; i < n; i += blockDim.x * gridDim.x)
    {
        vec[i] = alpha;
    }
    printf("%lf \n", alpha);
}

/**************************************************************
*****   Vector Set Random Values
**************************************************************
***** Initializes each element of the vector to a random
***** value
**************************************************************/
__global__ void set_rand_values_kernel(int seed, data_t* vec, int n)
{
    // set seed using clock information

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i=tid; i < n; i += blockDim.x * gridDim.x)
    {
        curandState state;
        curand_init(clock64(), i, 0, &state);
        vec[i] = curand_uniform(&state); 
    }
}

/**************************************************************
*****   Vector Copy
**************************************************************
***** Copies on device portion of a vector to on device portion
***** of new vector
*****
***** Parameters
***** -------------
***** vec : data_t* 
*****    On device vector data to be copied.
***** new_vec : data_t* 
*****    On device vector data containing copy.
***** n : int 
*****    Length of vector.
**************************************************************/
__global__ void copy_kernel(data_t* vec, data_t* new_vec, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i=tid; i < n; i += blockDim.x * gridDim.x)
    {
        new_vec[i] = vec[i];
    }
}

/**************************************************************
*****   Vector Append
**************************************************************
***** Appends on device data held in vec2 to vec1, storing
***** result in new_vec
*****
***** Parameters 
***** ------------
***** vec1 : data_t*
*****    On device vector data being appended to.
***** vec2 : data_t*
*****    On device vector data being appended.
***** new_vec : data_t*
*****    On device vector data containing result.
***** n : int
*****    Length of vec1 and vec2.
**************************************************************/
__global__ void append_kernel(data_t* vec1, data_t* vec2, data_t* new_vec, int n)
{
    // SHOULD HAVE 3 VECTORS AS INPUT? AND THEN COPY THE FIRST
	// 2 VECTORS INTO THE NEW THIRD ONE? 
    // THEN REPLACE THE DEVICE POINTER FOR THE VECTOR WITH THE 
	// NEW VEC POINTER AND DEALLOCATED THE FIRST VECTOR
    // STORE NEW VALUES IN APPROPRIATE PLACE ON DEVICE
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i=tid; i < n; i += blockDim.x * gridDim.x)
    {
        new_vec[i] = vec1[i];
        new_vec[n+i] = vec2[i];
    }
}

/**************************************************************
*****   Vector Split 
**************************************************************
***** Splits the vector into t b_vecs
*****
***** Parameters 
***** ------------
***** vec : data_t*
*****    The on device vector to split.
***** new_vec : data_t*
*****    The on device vector to contain the split of vec.
***** t : int
*****    The number of b_vecs to split vec into.
***** n : int
*****    The length of vec - used in determining where to
*****    store values in split vector.
**************************************************************/
__global__ void split_kernel(data_t* vec, data_t* new_vec, int t, int n)
{
    // This split was designed for distributed memory machine with mpi
	// for single node cuda implementation, this just results in an
	// on node copy of the vector of the device
    // ALLOCATE NEW ON DEVICE VECTOR OF APPROPRIATE SIZE
    // STORE NEW VALUES IN APPROPRIATE PLACE IN NEW VEC
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i=tid; i < n; i += blockDim.x * gridDim.x)
    {
        new_vec[i] = vec[i];
    }

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
***** vec : data_t*
*****    The on device vector to split.
***** new_vec : data_t*
*****    The on device vector to contain the split of vec.
***** t : int
*****    The number of b_vecs to split vec into
***** start : int
*****    The index of new_vec that should contain the
*****    first portion of the vec's values.
**************************************************************/
__global__ void split_range_kernel(data_t* vec, data_t* new_vec, int t, int start)
{
    // RESIZE ON DEVICE VECTOR
    // STORE NEW VALUES IN APPROPRIATE PLACE ON DEVICE

    // new vec already resized to fit all vectors 

    /*W.b_vecs = t;
    W.resize(num_values);
    W.set_const_value(0.0);

    for (int i = 0; i < num_values; i++)
    {
        W.values[start*num_values + i] = values[i];
        start = (start + 1) % t;
    }*/
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
***** vec : data_t*
*****    The on device vector to split.
***** new_vec : data_t*
*****    The on device vector to contain the split of vec.
***** t : int
*****    The number of b_vecs to split vec into
***** first_global_index : int
*****    The corresponding global index of the first index
*****    in this vector 
**************************************************************/
__global__ void split_contig_kernel(data_t* vec, data_t* new_vec, int t, int first_global_index)
{
    // RESIZE ON DEVICE VECTOR
    // STORE NEW VALUES IN APPROPRIATE PLACE ON DEVICE
    
    /*int glob_index, bvec, pos_in_bvec, end;
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
    }*/
}
#endif
