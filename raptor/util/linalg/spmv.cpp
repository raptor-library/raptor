// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/matrix.hpp"

using namespace raptor;

// COOMatrix SpMV Methods (or BCOO)
void COOMatrix::spmv_append(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        append(&b[idx1[i]], &x[idx2[i]], vals[i]);
    }
}
void COOMatrix::spmv_append_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        append_T(&b[idx2[i]], &x[idx1[i]], vals[i]);
    }
}
void COOMatrix::spmv_append_neg(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        append_neg(&b[idx1[i]], &x[idx2[i]], vals[i]);
    }
}
void COOMatrix::spmv_append_neg_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    for (int i = 0; i < nnz; i++)
    {
        append_neg_T(&b[idx2[i]], &x[idx1[i]], vals[i]);
    }
}


// CSRMatrix SpMV Methods (or BSR)
void CSRMatrix::spmv_append(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append(&b[i], &x[idx2[j]], vals[j]);
        }
    }
}
void CSRMatrix::spmv_append_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append_T(&b[idx2[j]], &x[i], vals[j]);
        }
    }
}
void CSRMatrix::spmv_append_neg(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append_neg(&b[i], &x[idx2[j]], vals[j]);
        }
    }
}
void CSRMatrix::spmv_append_neg_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append_neg_T(&b[idx2[j]], &x[i], vals[j]);
        }
    }
}



// CSCMatrix SpMV Methods (or BSC)
void CSCMatrix::spmv_append(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append(&b[idx2[j]], &x[i], vals[j]);
        }
    }
}
void CSCMatrix::spmv_append_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append_T(&b[i], &x[idx2[j]], vals[j]);
        }
    }
}
void CSCMatrix::spmv_append_neg(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append_neg(&b[idx2[j]], &x[i], vals[j]);
        }
    }
}
void CSCMatrix::spmv_append_neg_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            append_neg_T(&b[i], &x[idx2[j]], vals[j]);
        }
    }
}

