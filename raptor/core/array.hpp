// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "types.hpp"
#include <stdlib.h>
#include <cassert>

namespace raptor
{

/**************************************************************
*****   Compare Method
**************************************************************
***** Compares two values (for qsort)
**************************************************************/
template <typename T>
index_t compare(const void* a, const void* b)
{
    const T* ia = (const T*) a;
    const T* ib = (const T*) b;
    return *ia - *ib;
}

/**************************************************************
 *****   Array Class
 **************************************************************
 ***** This class constructs a Array object, similar to std::vector
 ***** This class allows for creation of array from a data pointer
 *****
 ***** Attributes
 ***** -------------
 ***** data_ptr : T*
 *****    Pointer to data stored in Array
 ***** n : index_t
 *****    Size of array
 ***** alloc_n : index_t
 *****    Allocation of array (often different than size so that
 *****    not re-allocated each time an etry is added.
 ***** 
 ***** Methods
 ***** -------
 ***** reserve()
 *****    Reserve size for pointer (allocate space and set value
 *****    for alloc_n
 ***** resize()
 *****    Change the size.  Re-allocate data_ptr, set alloc_n,
 *****    and set size of n.
 ***** push_back()
 *****    Add value to end of array
 ***** set_data()
 *****    Set data in array from a pointer (shallow copy).
 ***** sort()
 *****    Sorts the data with qsort
 ***** data()
 *****    Returns pointer to data
 ***** size()
 *****    Returns size of array
 ***** TODO -- do we comment on T& operator[] ?
 **************************************************************/

template <typename T>
class Array
{

public:
    /**************************************************************
    *****   Array Class Constructor
    **************************************************************
    ***** Initializes an empty Array object
    **************************************************************/
    Array()
    {
        data_ptr = (T*) calloc (2, sizeof(T));
        n = 0;
        alloc_n = 2;
    }

    /**************************************************************
    *****   Array Class Constructor
    **************************************************************
    ***** Initializes an empty Array object, reserving a 
    ***** given size
    *****
    ***** Paramters
    ***** ------------
    ***** _n : index_t
    *****    Size to reserve 
    **************************************************************/
    Array(index_t _n)
    {
        data_ptr = (T*) calloc (_n, sizeof(T));
        n = 0;
        alloc_n = _n;
    }

    /**************************************************************
    *****   Array Class Constructor
    **************************************************************
    ***** Initializes an Array object, setting the data to 
    ***** that passed as a parameter
    *****
    ***** Paramters
    ***** ------------
    ***** _n : index_t
    *****    Size of data_ptr
    ***** _data_ptr : T*
    *****    Pointer to set data to
    **************************************************************/
    Array(index_t _n, T* _data_ptr)
    {
        data_ptr = _data_ptr;
        n = _n;
        alloc_n = _n;
    }

    /**************************************************************
    *****   Array Class Destructor
    **************************************************************
    ***** Denitializes the Array class, freeing data
    **************************************************************/
    ~Array()
    {
        if (alloc_n > 0)
        {
            free(data_ptr);
        }
    }

    /**************************************************************
    *****   Array Reserve Size
    **************************************************************
    ***** Allocates the data pointer to the passed size
    *****
    ***** Paramters
    ***** ------------
    ***** _n : index_t
    *****    Size to allocate 
    **************************************************************/
    void reserve(const index_t _n)
    {
        data_ptr = (T*) realloc(data_ptr, _n*sizeof(T));
        alloc_n = _n;
    }

    /**************************************************************
    *****   Array Resize
    **************************************************************
    ***** Resizes the array to the passed size
    *****
    ***** Paramters
    ***** ------------
    ***** _n : index_t
    *****    Size to make array 
    **************************************************************/
    void resize(const index_t _n)
    {
        reserve(_n);
        n = _n;
    }

    /**************************************************************
    *****   Array Push Back
    **************************************************************
    ***** Add value to array
    *****
    ***** Paramters
    ***** ------------
    ***** element : T
    *****    element to add to the end of the Array
    **************************************************************/
    void push_back(T element)
    {
        if (n == alloc_n)
        {
            data_ptr = (T*) realloc(data_ptr, 2*n*sizeof(T));
            alloc_n *= 2;
        }
        data_ptr[n++] = element;
    }
  
    /**************************************************************
    *****   Array Set Data
    **************************************************************
    ***** Set data in Array equal to a pointer of data
    ***** (Shallow Copy Data)
    *****
    ***** Paramters
    ***** ------------
    ***** _n : index_t
    *****    Size of passed data pointer
    ***** _data_ptr : T*
    *****    Pointer to data, from which to set Array data
    **************************************************************/
    void set_data(index_t _n, T* _data_ptr)
    {
        if (alloc_n > 0)
        {
            free(data_ptr);
        }
        data_ptr = _data_ptr;
        n = _n;
        alloc_n = _n;
    }

    /**************************************************************
    *****   Array Sort
    **************************************************************
    ***** Sort the entire array
    **************************************************************/
    void sort()
    {
        sort(0, n);
    }

    /**************************************************************
    *****   Array Sort (Partial)
    **************************************************************
    ***** Sort the array at all positions between the parameters
    ***** (Exclusive)
    *****
    ***** Paramters
    ***** ------------
    ***** start : index_t
    *****    Initial position of array to start sorting
    ***** end : index_t 
    *****    End of portion to be sorted (exclusive)
    **************************************************************/
    void sort(index_t start, index_t end)
    {
        const index_t n_sort = end - start;
        qsort(&data_ptr[start], n_sort, sizeof(T), compare<T>);
    }
 
    /**************************************************************
    *****   Array Get Data
    **************************************************************
    ***** Return pointer to data
    *****
    ***** Returns
    ***** ------------
    ***** T* : pointer to data
    **************************************************************/
    T* data()
    {
        return data_ptr;
    }

    /**************************************************************
    *****   Array Get Size
    **************************************************************
    ***** Return size of data
    *****
    ***** Returns
    ***** ------------
    ***** index_t : size of Array
    **************************************************************/
    index_t size()
    {
        return n;
    }

    /**************************************************************
    *****   Array Element Access
    **************************************************************
    ***** Function overload for element access
    *****
    ***** Returns
    ***** ------------
    ***** T& element at position passed
    **************************************************************/
    T& operator[](const index_t index)
    {
        assert(index >= 0 && index < n);
        return data_ptr[index];
    }

    index_t n;
    index_t alloc_n;
    T* data_ptr;

};

}


#endif
