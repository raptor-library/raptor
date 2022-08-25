// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/matrix.hpp"

using namespace raptor;

// Helper function - computes LU factorization in place in LU
void gauss_elim(const CSCMatrix* A, double* LU)
{
    int col_start, col_end, row;

    // Convert A to dense and store in LU
    for (int j = 0; j < A->n_cols; j++)
    {
        // Zero out column
        for (int i = 0; i < A->n_rows; i++)
        {
            LU[i*A->n_rows + j] = 0.0; 
        }
        // Fill non-zero entries of column
        col_start = A->idx1[j];
        col_end   = A->idx1[j+1];
        for (int i = col_start; i < col_end; i++)
        {
            row = A->idx2[i]; 
            LU[row*A->n_rows + j] = A->vals[i];
        }
    }

    // Loop over columns
    for (int k = 0; k < (A->n_cols-1); k++)
    {
        // Stop if pivot is zero
        //if (fabs(LU[k*A->n_rows + k]) < 1e-16) printf("zero pivot\n");
        // Compute multipliers
        for (int i = (k+1); i < A->n_rows; i++)
        {
            LU[i*A->n_rows + k] /= LU[k*A->n_rows + k];
        }
        // Apply transformation to remaining submatrix
        for (int j = (k+1); j < A->n_cols; j++)
        {
            for (int i = (k+1); i < A->n_cols; i++)
            {
                LU[i*A->n_rows + j] -= LU[i*A->n_rows + k] * LU[k*A->n_rows + j];
            }
        }
    }
}

// Helper function - forward substitution - assumes matrix is nonsingular
void forward_sub(const CSCMatrix* L, double* x, double* y, const int nvecs)
{
    int col_start, col_end, row;
    for (int j = 0; j < L->n_cols; j++)
    {
        col_start = L->idx1[j];
        col_end   = L->idx1[j+1];
        // Calculate solution component
        for (int k = 0; k < nvecs; k++)
        {
            x[k*L->n_rows + j] = y[k*L->n_rows + j] / L->vals[col_start];
        }
        // Loop over rows and update rhs
        for (int i = col_start+1; i < col_end; i++)
        {
            row = L->idx2[i];
            for (int k = 0; k < nvecs; k++)
            {
                y[k*L->n_rows + row] -= L->vals[i] * x[k*L->n_rows + j];
            }
        }
    }
}

// Helper function - backward substitution - assumes matrix is nonsingular
void backward_sub(const CSCMatrix* U, double* x, double* y, const int nvecs)
{
    int col_start, col_end, row;
    for (int j = (U->n_cols-1); j >= 0; j--)
    {
        col_start = U->idx1[j];
        col_end   = U->idx1[j+1];
        // Calculate solution component
        for (int k = 0; k < nvecs; k++)
        {
            x[k*U->n_rows + j] = y[k*U->n_rows + j] / U->vals[col_end-1];
        }
        // Loop over rows and update rhs
        for (int i = col_start; i < col_end-1; i++)
        {
            row = U->idx2[i];
            for (int k = 0; k < nvecs; k++)
            {
                y[k*U->n_rows + row] -= U->vals[i] * x[k*U->n_rows + j];
            }
        }
    }
}



// Matrix functions
// Assumes A is non-singular
void Matrix::gaussian_elimination(CSRMatrix* L, CSRMatrix* U)
{
    CSCMatrix* A = to_CSC();
    double* LU = new double[A->n_rows * A->n_cols];
    gauss_elim(A, LU);

    // Extract L
    L->idx1.resize(L->n_rows + 1);
    L->nnz = 0;
    L->idx1[0] = 0;
    for (int i = 0; i < L->n_rows; i++)
    {
        L->idx2.push_back(i);
        L->vals.push_back(1.0);
        L->nnz++;
        // Add subdiagonal entries
        for (int j = 0; j < i; j++)
        {
            int pos = i * L->n_cols + j;
            if (fabs(LU[pos]))
            {
                L->idx2.push_back(j);
                L->vals.push_back(LU[pos]);
                L->nnz++;
            }
        }
        L->idx1[i+1] = L->nnz;
    }

    // Extract U
    U->idx1.resize(U->n_rows + 1);
    U->nnz = 0;
    U->idx1[0] = 0;
    for (int i = 0; i < U->n_rows; i++)
    {
        // Add subdiagonal entries
        for (int j = i; j < U->n_cols; j++)
        {
            int pos = i * U->n_cols + j;
            if (fabs(LU[pos]))
            {
                U->idx2.push_back(j);
                U->vals.push_back(LU[pos]);
                U->nnz++;
            }
        }
        U->idx1[i+1] = U->nnz;
    }

    delete LU;
}

// Assumes bvecs are the same in x and b
void Matrix::forward_substitution(Vector &x, Vector&b)
{
    CSCMatrix* L = to_CSC();
    L->sort();
    std::vector<double> b_copy;
    std::copy(b.values.begin(), b.values.end(), std::back_inserter(b_copy));
    forward_sub(L, x.data(), b_copy.data(), x.b_vecs);
}

// Assumes bvecs are the same in x and b
void Matrix::backward_substitution(Vector &x, Vector&b)
{
    CSCMatrix* U = to_CSC();
    U->sort();
    std::vector<double> b_copy; 
    std::copy(b.values.begin(), b.values.end(), std::back_inserter(b_copy));
    backward_sub(U, x.data(), b_copy.data(), x.b_vecs);
}



