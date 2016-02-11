// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef ARRAY_HPP
#define ARRAY_HPP

#include "types.hpp"
#include <stdlib.h>
#include <cassert>

namespace raptor
{

template <typename T>
index_t compare(const void* a, const void* b)
{
    const T* ia = (const T*) a;
    const T* ib = (const T*) b;
    return *ia - *ib;
}

template <typename T>
class Array
{

public:
    Array()
    {
        data_ptr = (T*) calloc (2, sizeof(T));
        n = 0;
        alloc_n = 2;
    }

    Array(index_t _n)
    {
        data_ptr = (T*) calloc (_n, sizeof(T));
        n = 0;
        alloc_n = _n;
    }

    Array(index_t _n, T* _data_ptr)
    {
        data_ptr = _data_ptr;
        n = _n;
        alloc_n = _n;
    }

    ~Array()
    {
        if (alloc_n > 0)
        {
            free(data_ptr);
        }
    }

    void reserve(const index_t _n)
    {
        data_ptr = (T*) realloc(data_ptr, _n*sizeof(T));
        alloc_n = _n;
    }

    void resize(const index_t _n)
    {
        reserve(_n);
        n = _n;
    }

    void push_back(T element)
    {
        if (n == alloc_n)
        {
            data_ptr = (T*) realloc(data_ptr, 2*n*sizeof(T));
            alloc_n *= 2;
        }
        data_ptr[n++] = element;
    }
  
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

    void sort()
    {
        sort(0, n);
    }

    void sort(index_t start, index_t end)
    {
        const index_t n_sort = end - start;
        qsort(&data_ptr[start], n_sort, sizeof(T), compare<T>);
    }
 
    T* data()
    {
        return data_ptr;
    }

    index_t size()
    {
        return n;
    }

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
