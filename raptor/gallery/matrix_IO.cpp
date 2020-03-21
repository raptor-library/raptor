// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "matrix_IO.hpp"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

bool little_endian()
{
    int num = 1;
    return (*(char *)&num == 1);
}

template <class T>
void endian_swap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

CSRMatrix* readMatrix(const char* filename)
{
    CSRMatrix* A;

    int32_t code;
    int32_t n_rows;
    int32_t n_cols;
    int32_t nnz;
    int32_t idx;
    double val;

    int sizeof_dbl = sizeof(val);
    int sizeof_int32 = sizeof(code);
    bool is_little_endian = false;

    std::ifstream ifs (filename, std::ifstream::binary);
    ifs.read(reinterpret_cast<char *>(&code), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&n_rows), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&n_cols), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&nnz), sizeof_int32);

    if (code != PETSC_MAT_CODE)
    {
        is_little_endian = true;
        endian_swap(&code);
        endian_swap(&n_rows);
        endian_swap(&n_cols);
        endian_swap(&nnz);
    }

    assert(code == PETSC_MAT_CODE);

    A = new CSRMatrix(n_rows, n_cols, nnz);

    int displ = 0;
    A->idx1[0] = 0;
    if (is_little_endian)
    {
        for (int32_t i = 0; i < n_rows; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            endian_swap(&idx);
            displ += idx;
            A->idx1[i+1] = displ;
        }
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            endian_swap(&idx);
            A->idx2.emplace_back(idx);
        }
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&val), sizeof_dbl);
            endian_swap(&val);
            A->vals.emplace_back(val);
        }
    }
    else
    {
        for (int32_t i = 0; i < n_rows; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            displ += idx;
            A->idx1[i+1] = displ;
        }   
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            A->idx2.emplace_back(idx);
        }
        for (int32_t i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&val), sizeof_dbl);
            endian_swap(&val);
            A->vals.emplace_back(val);
        }
    }
    A->nnz = A->idx2.size();

    ifs.close();

    return A;
    
}


