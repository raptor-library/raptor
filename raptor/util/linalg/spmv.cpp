// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/matrix.hpp"

using namespace raptor;

// COOMatrix SpMV Methods (or BCOO)
template <typename T>
void COO_append(const COOMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append(A->idx1[i], A->idx2[i], b.data(), x.data(), vals[i]);
    }
}
template <typename T>
void COO_append_T(const COOMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append_T(A->idx2[i], A->idx1[i], b.data(), x.data(), vals[i]);
    }
}
template <typename T>
void COO_append_neg(const COOMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append_neg(A->idx1[i], A->idx2[i], b.data(), x.data(), vals[i]);
    }
}
template <typename T>
void COO_append_neg_T(const COOMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append_neg_T(A->idx1[i], A->idx2[i], b.data(), x.data(), vals[i]);
    }
}

// CSRMatrix SpMV Methods (or BSR)
template <typename T>
void CSR_append(const CSRMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append(i, A->idx2[j], b.data(), x.data(), vals[j]);
        }
    }
}
template <typename T>
void CSR_append_T(const CSRMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_T(i, A->idx2[j], b.data(), x.data(), vals[j]);
        }
    }
}
template <typename T>
void CSR_append_neg(const CSRMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_neg(i, A->idx2[j], b.data(), x.data(), vals[j]);
        }
    }
}
template <typename T>
void CSR_append_neg_T(const CSRMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_neg_T(i, A->idx2[j], b.data(), x.data(), vals[j]);
        }
    }
}



// CSCMatrix SpMV Methods (or BSC)
template <typename T>
void CSC_append(const CSCMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append(A->idx2[j], i, b.data(), x.data(), vals[j]);
        }
    }
}
template <typename T>
void CSC_append_T(const CSCMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_T(A->idx2[j], i, b.data(), x.data(), vals[j]);
        }
    }
}
template <typename T>
void CSC_append_neg(const CSCMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_neg(A->idx2[j], i, b.data(), x.data(), vals[j]);
        }
    }
}
template <typename T>
void CSC_append_neg_T(const CSCMatrix* A, const aligned_vector<T>& vals,
        const aligned_vector<double>& x, aligned_vector<double>& b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_neg_T(A->idx2[j], i, b.data(), x.data(), vals[j]);
        }
    }
}





void COOMatrix::spmv_append(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    COO_append(this, vals, x, b);
}
void COOMatrix::spmv_append_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    COO_append_T(this, vals, x, b);
}
void COOMatrix::spmv_append_neg(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    COO_append_neg(this, vals, x, b);
}
void COOMatrix::spmv_append_neg_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    COO_append_neg_T(this, vals, x, b);
}
void BCOOMatrix::spmv_append(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    COO_append(this, vals, x, b);
}

void BCOOMatrix::spmv_append_T(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    COO_append_T(this, vals, x, b);
}

void BCOOMatrix::spmv_append_neg(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    COO_append_neg(this, vals, x, b);
}

void BCOOMatrix::spmv_append_neg_T(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    COO_append_neg_T(this, vals, x, b);
}



void CSRMatrix::spmv_append(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSR_append(this, vals, x, b);
}
void CSRMatrix::spmv_append_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSR_append_T(this, vals, x, b);
}
void CSRMatrix::spmv_append_neg(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSR_append_neg(this, vals, x, b);
}
void CSRMatrix::spmv_append_neg_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSR_append_neg_T(this, vals, x, b);
}
void BSRMatrix::spmv_append(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSR_append(this, vals, x, b);
}

void BSRMatrix::spmv_append_T(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSR_append_T(this, vals, x, b);
}

void BSRMatrix::spmv_append_neg(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSR_append_neg(this, vals, x, b);
}

void BSRMatrix::spmv_append_neg_T(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSR_append_neg_T(this, vals, x, b);
}




void CSCMatrix::spmv_append(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSC_append(this, vals, x, b);
}
void CSCMatrix::spmv_append_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSC_append_T(this, vals, x, b);
}
void CSCMatrix::spmv_append_neg(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSC_append_neg(this, vals, x, b);
}
void CSCMatrix::spmv_append_neg_T(const aligned_vector<double>& x, 
        aligned_vector<double>& b) const
{
    CSC_append_neg_T(this, vals, x, b);
}
void BSCMatrix::spmv_append(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSC_append(this, vals, x, b);
}
void BSCMatrix::spmv_append_T(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSC_append_T(this, vals, x, b);
}
void BSCMatrix::spmv_append_neg(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSC_append_neg(this, vals, x, b);
}
void BSCMatrix::spmv_append_neg_T(const aligned_vector<double>& x,
        aligned_vector<double>& b) const
{
    CSC_append_neg_T(this, vals, x, b);
}



