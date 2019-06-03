// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/matrix.hpp"

using namespace raptor;

// COOMatrix SpMV Methods (or BCOO)
template <typename T>
void COO_append(const COOMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append(A->idx1[i], A->idx2[i], b, x, vals[i]);
    }
}
template <typename T>
void COO_append_T(const COOMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append_T(A->idx2[i], A->idx1[i], b, x, vals[i]);
    }
}
template <typename T>
void COO_append_neg(const COOMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append_neg(A->idx1[i], A->idx2[i], b, x, vals[i]);
    }
}
template <typename T>
void COO_append_neg_T(const COOMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    for (int i = 0; i < A->nnz; i++)
    {
        A->append_neg_T(A->idx1[i], A->idx2[i], b, x, vals[i]);
    }
}





// CSRMatrix SpMV Methods (or BSR)
// Optimized CSR and BSR standard SpMVs
void CSR_spmv(const CSRMatrix* A, const double* x, double* b, const int n_vecs = 1)
{
    int start, end;
    double val;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        int v;
        #pragma omp parallel for default(none) private(v, val) \
            shared(start, end, i, A, x, b) schedule(static)
        for (v = 0; v < n_vecs; v++)
        {
            val = 0;
            for (int j = start; j < end; j++)
            {
                val += A->vals[j] * x[A->idx2[j] + v*A->n_cols];
            }
            b[i + v*A->n_rows] = val;
        }
    }
}
/*void CSR_spmv(const CSRMatrix* A, const double* x, double* b)
{
    int start, end;
    double val;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        val = 0;
        for (int j = start; j < end; j++)
        {
            val += A->vals[j] * x[A->idx2[j]];
        }
        b[i] = val;
    }
}*/

void CSR_residual(const CSRMatrix* A, const double* x, 
        const double* b, double* r, const int n_vecs = 1)
{
    int start, end;
    double val;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        int v;
        /*#pragma omp parallel for default(none) private(v, val) \
            shared(start, end, i, A, x, b, r) schedule(static)*/
        for (v = 0; v < n_vecs; v++)
        {
            val = b[i + v*A->n_rows];
            for (int j = start; j < end; j++)
            {
                val -= A->vals[j] * x[A->idx2[j] + v*A->n_cols];
            }
            r[i + v*A->n_rows] = val;
        }
    }
}


void CSR_append(const CSRMatrix* A, const double* x, double* b, const int n_vecs = 1, const int xlen = 0)
{
    int start, end;
    double val;
    int v_offset = xlen / n_vecs;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        int v;
        #pragma omp parallel for default(none) private(v, val) \
            shared(start, end, i, A, x, b, v_offset) schedule(static)
        for (v = 0; v < n_vecs; v++)
        {
            val = 0;
            for (int j = start; j < end; j++)
            {
                val += A->vals[j] * x[A->idx2[j] + v*v_offset];
            }
            b[i + v*A->n_rows] += val;
        }
    }
}

/*void CSR_append(const CSRMatrix* A, const double* x, double* b)
{
    int start, end;
    double val;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        val = 0;
        for (int j = start; j < end; j++)
        {
            val += A->vals[j] * x[A->idx2[j]];
        }
        b[i] += val;
    }
}*/

template <typename T>
void BSR_append(const CSRMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append(i, A->idx2[j], b, x, vals[j]);
        }
    }
}

void BSR_spmv(const BSRMatrix* A, const double* x, double* b, const int n_vecs = 1)
{
    int start, end, idx;
    int first_row, first_col;
    double val;
    double* block_val;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        first_row = i*A->b_rows;
        for (int row = 0; row < A->b_rows; row++)
        {
            idx = row * A->b_cols;
            for (int v = 0; v < n_vecs; v++)
            {
                val = 0;
                for (int j = start; j < end; j++)
                {
                    first_col = A->idx2[j]*A->b_cols;
                    block_val = A->block_vals[j];
                    for (int col = 0; col < A->b_cols; col++)
                    {
                        val += (block_val[idx + col] * x[first_col + col + v*(A->n_cols*A->b_cols)]);
                    }
                }
                b[first_row + row + v*(A->n_rows*A->b_rows)] = val;
            }
        }
    }
}
template <typename T>
void CSR_append_T(const CSRMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b, const int n_vecs = 1, const int blen = 0)
{
    int start, end;
    int v_offset = blen / n_vecs;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int v = 0; v < n_vecs; v++)
        {
            for (int j = start; j < end; j++)
            {
                A->append_T(i + v*A->n_cols, v*v_offset + A->idx2[j], b, x, vals[j]);
            }
        }
    }
}
template <typename T>
void CSR_append_neg(const CSRMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b, const int n_vecs = 1)
{
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        int v;
        #pragma omp parallel for default(none) private(v) \
            shared(start, end, i, A, x, b, vals) schedule(static)
        for (v = 0; v < n_vecs; v++)
        {
            for (int j = start; j < end; j++)
            {
                A->append_neg(i + v*A->n_rows, v*A->n_cols + A->idx2[j], b, x, vals[j+v*A->n_cols]);
            }
        }
    }
}
template <typename T>
void CSR_append_neg_T(const CSRMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b, const int n_vecs = 1)
{
    printf("negative transpose called\n");
    int start, end;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int v = 0; v < n_vecs; v++)
        {
            for (int j = start; j < end; j++)
            {
                //A->append_neg(i + v*A->n_rows, v*A->n_cols + A->idx2[j], b, x, vals[j+v*A->n_cols]);
                A->append_neg_T(i + v*A->n_rows, v*A->n_cols + A->idx2[j], b, x, vals[j+v*A->n_cols]);
            }
        }
    }
}



// CSCMatrix SpMV Methods (or BSC)
template <typename T>
void CSC_append(const CSCMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append(A->idx2[j], i, b, x, vals[j]);
        }
    }
}
template <typename T>
void CSC_append_T(const CSCMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_T(A->idx2[j], i, b, x, vals[j]);
        }
    }
}
template <typename T>
void CSC_append_neg(const CSCMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_neg(A->idx2[j], i, b, x, vals[j]);
        }
    }
}
template <typename T>
void CSC_append_neg_T(const CSCMatrix* A, const aligned_vector<T>& vals,
        const double* x, double* b)
{
    int start, end;
    for (int i = 0; i < A->n_cols; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            A->append_neg_T(A->idx2[j], i, b, x, vals[j]);
        }
    }
}


void COOMatrix::spmv(const double* x, double* b, const int n_vecs) const
{
    for (int i = 0; i < n_rows; i++)
        b[i] = 0;
    COO_append(this, vals, x, b);
}
void COOMatrix::spmv_append(const double* x, double* b, const int n_vecs, const int xlen) const
{
    COO_append(this, vals, x, b);
}
void COOMatrix::spmv_append_T(const double* x, double* b, const int n_vecs, const int blen) const
{
    COO_append_T(this, vals, x, b);
}
void COOMatrix::spmv_append_neg(const double* x, double* b, const int n_vecs) const
{
    COO_append_neg(this, vals, x, b);
}
void COOMatrix::spmv_append_neg_T(const double* x, double* b, const int n_vecs) const
{
    COO_append_neg_T(this, vals, x, b);
}
void COOMatrix::spmv_residual(const double* x, const double* b, double* r, const int n_vecs) const
{
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];
    COO_append_neg(this, vals, x, r);
}
void BCOOMatrix::spmv(const double* x, double* b, const int n_vecs) const 
{
    for (int i = 0; i < n_rows * b_rows; i++)
        b[i] = 0;
    COO_append(this, block_vals, x, b);
}
void BCOOMatrix::spmv_append(const double* x,double* b, const int n_vecs, const int xlen) const
{
    COO_append(this, block_vals, x, b);
}
void BCOOMatrix::spmv_append_T(const double* x,double* b, const int n_vecs, const int blen) const
{
    COO_append_T(this, block_vals, x, b);
}
void BCOOMatrix::spmv_append_neg(const double* x,double* b, const int n_vecs) const
{
    COO_append_neg(this, block_vals, x, b);
}
void BCOOMatrix::spmv_append_neg_T(const double* x,double* b, const int n_vecs) const
{
    COO_append_neg_T(this, block_vals, x, b);
}
void BCOOMatrix::spmv_residual(const double* x, const double* b, double* r, const int n_vecs) const
{
    for (int i = 0; i < n_rows * b_rows; i++)
        r[i] = b[i];
    COO_append_neg(this, block_vals, x, r);
}



void CSRMatrix::spmv(const double* x, double* b, const int n_vecs) const
{
    CSR_spmv(this, x, b, n_vecs);
}
void CSRMatrix::spmv_append(const double* x, double* b, const int n_vecs, const int xlen) const
{
    CSR_append(this, x, b, n_vecs, xlen);
}
void CSRMatrix::spmv_append_T(const double* x, double* b, const int n_vecs, const int blen) const
{
    CSR_append_T(this, vals, x, b, n_vecs, blen);
}
void CSRMatrix::spmv_append_neg(const double* x, double* b, const int n_vecs) const
{
    CSR_append_neg(this, vals, x, b, n_vecs);
}
void CSRMatrix::spmv_append_neg_T(const double* x, double* b, const int n_vecs) const
{
    CSR_append_neg_T(this, vals, x, b, n_vecs);
}
void CSRMatrix::spmv_residual(const double* x, const double* b, double* r, const int n_vecs) const
{
    CSR_residual(this, x, b, r, n_vecs);
}
void BSRMatrix::spmv(const double* x, double* b, const int n_vecs) const
{
    BSR_spmv(this, x, b);
}
void BSRMatrix::spmv_append(const double* x,double* b, const int n_vecs, const int xlen) const
{
    BSR_append(this, block_vals, x, b);
}
void BSRMatrix::spmv_append_T(const double* x,double* b, const int n_vecs, const int blen) const
{
    CSR_append_T(this, block_vals, x, b);
}
void BSRMatrix::spmv_append_neg(const double* x,double* b, const int n_vecs) const
{
    CSR_append_neg(this, block_vals, x, b);
}
void BSRMatrix::spmv_append_neg_T(const double* x,double* b, const int n_vecs) const
{
    CSR_append_neg_T(this, block_vals, x, b);
}
void BSRMatrix::spmv_residual(const double* x, const double* b, double* r, const int n_vecs) const
{
    for (int i = 0; i < n_rows * b_rows; i++)
        r[i] = b[i];
    CSR_append_neg(this, block_vals, x, r);
}



void CSCMatrix::spmv(const double* x, double* b, const int n_vecs) const
{
    for (int i = 0; i < n_rows; i++)
        b[i] = 0;
    CSC_append(this, vals, x, b);
}
void CSCMatrix::spmv_append(const double* x, double* b, const int n_vecs, const int xlen) const
{
    CSC_append(this, vals, x, b);
}
void CSCMatrix::spmv_append_T(const double* x, double* b, const int n_vecs, const int blen) const
{
    CSC_append_T(this, vals, x, b);
}
void CSCMatrix::spmv_append_neg(const double* x, double* b, const int n_vecs) const
{
    CSC_append_neg(this, vals, x, b);
}
void CSCMatrix::spmv_append_neg_T(const double* x, double* b, const int n_vecs) const
{
    CSC_append_neg_T(this, vals, x, b);
}
void CSCMatrix::spmv_residual(const double* x, const double* b, double* r, const int n_vecs) const
{
    for (int i = 0; i < n_rows; i++)
        r[i] = b[i];
    CSC_append_neg(this, vals, x, r);
}
void BSCMatrix::spmv(const double* x, double* b, const int n_vecs) const
{ 
    for (int i = 0; i < n_rows * b_rows; i++)
        b[i] = 0;
    CSC_append(this, block_vals, x, b);
}
void BSCMatrix::spmv_append(const double* x,double* b, const int n_vecs, const int xlen) const
{
    CSC_append(this, block_vals, x, b);
}
void BSCMatrix::spmv_append_T(const double* x,double* b, const int n_vecs, const int blen) const
{
    CSC_append_T(this, block_vals, x, b);
}
void BSCMatrix::spmv_append_neg(const double* x,double* b, const int n_vecs) const
{
    CSC_append_neg(this, block_vals, x, b);
}
void BSCMatrix::spmv_append_neg_T(const double* x,double* b, const int n_vecs) const
{
    CSC_append_neg_T(this, block_vals, x, b);
}
void BSCMatrix::spmv_residual(const double* x, const double* b, double* r, const int n_vecs) const
{
    for (int i = 0; i < n_rows * b_rows; i++)
        r[i] = b[i];
    CSC_append_neg(this, block_vals, x, r);
}



