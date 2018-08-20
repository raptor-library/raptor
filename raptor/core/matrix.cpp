// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/matrix.hpp"

using namespace raptor;

/**************************************************************
*****  Matrix Print
**************************************************************
***** Print the nonzeros in the matrix, as well as the row
***** and column according to each nonzero
**************************************************************/
template <typename T>
void print_helper(const COOMatrix* A, const aligned_vector<T>& vals)
{
    int row, col;
    double val;

    printf("A->nnz %d\n", A->nnz);
    for (int i = 0; i < A->nnz; i++)
    {
        row = A->idx1[i];
        col = A->idx2[i];
        A->val_print(row, col, vals[i]);
    }
}
template <typename T>
void print_helper(const CSRMatrix* A, const aligned_vector<T>& vals)
{
    int col, start, end;

    for (int row = 0; row < A->n_rows; row++)
    {
        start = A->idx1[row];
        end = A->idx1[row+1];
        for (int j = start; j < end; j++)
        {
            col = A->idx2[j];
            A->val_print(row, col, vals[j]);
        }
    }
}
template <typename T>
void print_helper(const CSCMatrix* A, const aligned_vector<T>& vals)
{
    int row, start, end;

    for (int col = 0; col < A->n_cols; col++)
    {
        start = A->idx1[col];
        end = A->idx1[col+1];
        for (int j = start; j < end; j++)
        {
            row = A->idx2[j];
            A->val_print(row, col, vals[j]);
        }
    }
}
void COOMatrix::print()
{
    print_helper(this, vals);
}
void CSRMatrix::print()
{
    print_helper(this, vals);
}
void CSCMatrix::print()
{
    print_helper(this, vals);
}
void BCOOMatrix::print()
{
    printf("Printing...\n");
    print_helper(this, vals);
}
void BSRMatrix::print()
{
    print_helper(this, vals);
}
void BSCMatrix::print()
{
    print_helper(this, vals);
}

/**************************************************************
*****  Matrix Transpose
**************************************************************
***** Transpose the matrix, reversing rows and columns
***** Retain matrix type, and block structure if applicable
**************************************************************/
COOMatrix* COOMatrix::transpose()
{
    COOMatrix* T = new COOMatrix(n_rows, n_cols, idx2, idx1, vals);
    return T;
}

BCOOMatrix* BCOOMatrix::transpose()
{
    BCOOMatrix* T = new BCOOMatrix(b_rows, b_cols, n_rows, n_cols, idx2, idx1, vals);
    return T;
}

CSRMatrix* CSRMatrix::transpose()
{
    CSCMatrix* T_csc = new CSCMatrix(n_rows, n_cols, idx1, idx2, vals); 
    CSRMatrix* T = T_csc->to_CSR();
    delete T_csc;
    return T;
}

BSRMatrix* BSRMatrix::transpose()
{
    BSCMatrix* T_bsc = new BSCMatrix(b_rows, b_cols, n_rows, n_cols, idx1, idx2, vals);
    BSRMatrix* T = (BSRMatrix*) T_bsc->to_CSR();
    delete T_bsc;
    return T;
}

CSCMatrix* CSCMatrix::transpose()
{
    CSRMatrix* T_csr = new CSRMatrix(n_rows, n_cols, idx1, idx2, vals); 
    CSCMatrix* T = T_csr->to_CSC();
    delete T_csr;
    return T;
}
BSCMatrix* BSCMatrix::transpose()
{
    BSRMatrix* T_bsr = new BSRMatrix(b_rows, b_cols, n_rows, n_cols, idx1, idx2, vals); 
    BSCMatrix* T = (BSCMatrix*) T_bsr->to_CSC();
    delete T_bsr;
    return T;
}


/**************************************************************
*****   Matrix Resize
**************************************************************
***** Set the matrix dimensions to those passed as parameters
*****
***** Parameters
***** -------------
***** _nrows : int
*****    Number of rows in matrix
***** _ncols : int
*****    Number of cols in matrix
**************************************************************/
void Matrix::resize(int _n_rows, int _n_cols)
{
    n_rows = _n_rows;
    n_cols = _n_cols;
}

/**************************************************************
*****   Matrix Copy
**************************************************************
***** Copy matrix between any subset of matrix types 
*****
***** Parameters
***** -------------
***** Matrix* A : original matrix to copy (of some type)
**************************************************************/
template <typename T>
void COO_to_COO(const COOMatrix* A, COOMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.clear();
    B->idx2.clear();
    B_vals.clear();

    B->idx1.reserve(A->nnz);
    B->idx2.reserve(A->nnz);
    B_vals.reserve(A->nnz);
    for (int i = 0; i < A->nnz; i++)
    {
        B->idx1.push_back(A->idx1[i]);
        B->idx2.push_back(A->idx2[i]);
        B_vals.push_back(B->copy_val(A_vals[i]));
    }
}
template <typename T>
void CSR_to_COO(const CSRMatrix* A, COOMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.clear();
    B->idx2.clear();
    B_vals.clear();

    B->idx1.reserve(A->nnz);
    B->idx2.reserve(A->nnz);
    B_vals.reserve(A->nnz);
    for (int i = 0; i < A->n_rows; i++)
    {
        int row_start = A->idx1[i];
        int row_end = A->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            B->idx1.push_back(i);
            B->idx2.push_back(A->idx2[j]);
            B_vals.push_back(B->copy_val(A_vals[j]));
        }
    }
}
template <typename T>
void CSC_to_COO(const CSCMatrix* A, COOMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.clear();
    B->idx2.clear();
    B_vals.clear();

    B->idx1.reserve(A->nnz);
    B->idx2.reserve(A->nnz);
    B_vals.reserve(A->nnz);
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            B->idx1.push_back(A->idx2[j]);
            B->idx2.push_back(i);
            B_vals.push_back(B->copy_val(A_vals[j]));
        }
    }

}
template <typename T>
void COO_to_CSR(const COOMatrix* A, CSRMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.resize(B->n_rows + 1);
    std::fill(B->idx1.begin(), B->idx1.end(), 0);
    if (B->nnz)
    {
        B->idx2.resize(B->nnz);
        if (A->data_size())
            B_vals.resize(B->nnz);
    }

    // Calculate indptr
    for (int i = 0; i < B->nnz; i++)
    {
        int row = A->idx1[i];
        B->idx1[row+1]++;
    }
    for (int i = 0; i < B->n_rows; i++)
    {
        B->idx1[i+1] += B->idx1[i];
    }

    // Add indices and data
    aligned_vector<int> ctr;
    if (B->n_rows)
    {
            ctr.resize(B->n_rows, 0);
    }
    for (int i = 0; i < B->nnz; i++)
    {
        int row = A->idx1[i];
        int col = A->idx2[i];
        int index = B->idx1[row] + ctr[row]++;
        B->idx2[index] = col;
        if (A->data_size()) // Checking that matrix has values (not S)
        {
            B_vals[index] = B->copy_val(A_vals[i]);
        }
    }

}
template <typename T>
void CSR_to_CSR(const CSRMatrix* A, CSRMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.resize(A->n_rows + 1);
    B->idx2.resize(A->nnz);
    B_vals.resize(A->nnz);

    B->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        B->idx1[i+1] = A->idx1[i+1];
        int row_start = B->idx1[i];
        int row_end = B->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            B->idx2[j] = A->idx2[j];
            B_vals[j] = B->copy_val(A_vals[j]);
        }
    }

}
template <typename T>
void CSC_to_CSR(const CSCMatrix* A, CSRMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.clear();
    B->idx2.clear();
    B_vals.clear();

    // Resize vectors to appropriate dimensions
    B->idx1.resize(A->n_rows + 1);
    B->idx2.resize(A->nnz);
    if (A->data_size())
        B_vals.resize(A->nnz);

    // Create indptr, summing number times row appears in CSC
    for (int i = 0; i <= A->n_rows; i++) B->idx1[i] = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        B->idx1[A->idx2[i] + 1]++;
    }
    for (int i = 1; i <= A->n_rows; i++)
    {
        B->idx1[i] += B->idx1[i-1];
    }

    // Add values to indices and data
    aligned_vector<int> ctr(B->n_rows, 0);
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row = A->idx2[j];
            int idx = B->idx1[row] + ctr[row]++;
            B->idx2[idx] = i;
            if (A->data_size())
            {
                B_vals[idx] = B->copy_val(A_vals[j]);
            }
        }
    }

}
template <typename T>
void COO_to_CSC(const COOMatrix* A, CSCMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.resize(B->n_cols + 1);
    std::fill(B->idx1.begin(), B->idx1.end(), 0);
    if (B->nnz)
    {
        B->idx2.resize(B->nnz);
        if (A->data_size())
            B_vals.resize(B->nnz);
    }

    // Calculate indptr
    for (int i = 0; i < B->nnz; i++)
    {
        int col = A->idx1[i];
        B->idx1[col+1]++;
    }
    for (int i = 0; i < B->n_cols; i++)
    {
        B->idx1[i+1] += B->idx1[i];
    }

    // Add indices and data
    aligned_vector<int> ctr;
    if (B->n_cols)
    {
        ctr.resize(B->n_cols, 0);
    }
    for (int i = 0; i < B->nnz; i++)
    {
        int col = A->idx1[i];
        int row = A->idx2[i];
        int index = B->idx1[col] + ctr[col]++;
        B->idx2[index] = row;
        if (A->data_size()) // Checking that matrix has values (not S)
        {
            B_vals[index] = B->copy_val(A_vals[i]);
        }
    }

}
template <typename T>
void CSR_to_CSC(const CSRMatrix* A, CSCMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.clear();
    B->idx2.clear();
    B_vals.clear();

    // Resize vectors to appropriate dimensions
    B->idx1.resize(A->n_cols + 1);
    B->idx2.resize(A->nnz);
    if (A->data_size())
        B_vals.resize(A->nnz);

    // Create indptr, summing number times row appears in CSC
    for (int i = 0; i <= A->n_cols; i++) B->idx1[i] = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        B->idx1[A->idx2[i] + 1]++;
    }
    for (int i = 1; i <= A->n_cols; i++)
    {
        B->idx1[i] += B->idx1[i-1];
    }

    // Add values to indices and data
    aligned_vector<int> ctr(B->n_cols, 0);
    for (int i = 0; i < A->n_rows; i++)
    {
        int row_start = A->idx1[i];
        int row_end = A->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = A->idx2[j];
            int idx = B->idx1[col] + ctr[col]++;
            B->idx2[idx] = i;
            if (A->data_size())
            {
                B_vals[idx] = B->copy_val(A_vals[j]);
            }
        }
    }

}
template <typename T>
void CSC_to_CSC(const CSCMatrix* A, CSCMatrix* B, aligned_vector<T>& A_vals,
        aligned_vector<T>& B_vals)
{
    B->n_rows = A->n_rows;
    B->n_cols = A->n_cols;
    B->nnz = A->nnz;

    B->idx1.resize(A->n_cols + 1);
    B->idx2.resize(A->nnz);
    B->vals.resize(A->nnz);

    B->idx1[0] = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        B->idx1[i+1] = col_end;
        for (int j = col_start; j < col_end; j++)
        {
            B->idx2[j] = A->idx2[j];
            B_vals[j] = B->copy_val(A_vals[j]);
        }
    }
}


/**************************************************************
*****   Matrix Sort
**************************************************************
***** Sorts the sparse matrix by row and column
**************************************************************/
template <typename T>
void sort_helper(COOMatrix* A, aligned_vector<T>& vals)
{
    if (A->sorted || A->nnz == 0)
    {
        A->sorted = true;
        return;
    }

    int k, prev_k;

    aligned_vector<int> permutation(A->nnz);
    aligned_vector<bool> done(A->nnz, false);

    // Create permutation vector p
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(),
        [&](int i, int j){ 
            if (A->idx1[i] == A->idx1[j])
                return A->idx2[i] < A->idx2[j];
            else
                return A->idx1[i] < A->idx1[j];
        });

    // Permute all vectors (rows, cols, data) 
    // according to p
    for (int i = 0; i < A->nnz; i++)
    {
        if (done[i]) continue;

        done[i] = true;
        prev_k = i;
        k = permutation[i];
        while (i != k)
        {
            std::swap(A->idx1[prev_k], A->idx1[k]);
            std::swap(A->idx2[prev_k], A->idx2[k]);
            std::swap(vals[prev_k], vals[k]);
            done[k] = true;
            prev_k = k;
            k = permutation[k];
        }
    }

    A->sorted = true;
    A->diag_first = false;

}

template <typename T>
void sort_helper(CSRMatrix* A, aligned_vector<T>& vals)
{
    int start, end, row_size;
    int k, prev_k;

    if (A->sorted || A->nnz == 0)
    {
        A->sorted = true;
        return;
    }

    aligned_vector<int> permutation;
    aligned_vector<bool> done;

    // Sort the columns of each row (and data accordingly) and remove
    // duplicates (summing values together)
    for (int row = 0; row < A->n_rows; row++)
    {
        start = A->idx1[row];
        end = A->idx1[row+1];
        row_size = end - start;
        if (row_size == 0) 
        {
            continue;
        }

        // Create permutation vector p for row
        permutation.resize(row_size);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(),
                [&](int i, int j)
                { 
                    return A->idx2[i+start] < A->idx2[j+start];
                });


        // Permute columns and data according to p
        done.resize(row_size);
        for (int i = 0; i < row_size; i++)
        {
            done[i] = false;
        }
        if (A->data_size())
        {
            for (int i = 0; i < row_size; i++)
            {
                if (done[i]) continue;

                done[i] = true;
                prev_k = i;
                k = permutation[i];
                while (i != k)
                {
                    std::swap(A->idx2[prev_k + start], A->idx2[k + start]);
                    std::swap(vals[prev_k + start], vals[k + start]);
                    done[k] = true;
                    prev_k = k;
                    k = permutation[k];
                }
            }
        }
        else
        {
            for (int i = 0; i < row_size; i++)
            {
                if (done[i]) continue;

                done[i] = true;
                prev_k = i;
                k = permutation[i];
                while (i != k)
                {
                    std::swap(A->idx2[prev_k + start], A->idx2[k + start]);
                    done[k] = true;
                    prev_k = k;
                    k = permutation[k];
                }
            }
        }
    }

    A->sorted = true;
    A->diag_first = false;
}

template <typename T>
void sort_helper(CSCMatrix* A, aligned_vector<T>& vals)
{
    int start, end, col_size;
    int k, prev_k;

    if (A->sorted || A->nnz == 0)
    {
        A->sorted = true;
        return;
    }

    aligned_vector<int> permutation;
    aligned_vector<bool> done;

    // Sort the columns of each col (and data accordingly) and remove
    // duplicates (summing values together)
    for (int col = 0; col < A->n_cols; col++)
    {
        start = A->idx1[col];
        end = A->idx1[col+1];
        col_size = end - start;
        if (col_size == 0) 
        {
            continue;
        }

        // Create permutation vector p for col
        permutation.resize(col_size);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(),
                [&](int i, int j)
                { 
                    return A->idx2[i+start] < A->idx2[j+start];
                });


        // Permute columns and data according to p
        done.resize(col_size);
        for (int i = 0; i < col_size; i++)
        {
            done[i] = false;
        }
        if (A->data_size())
        {
            for (int i = 0; i < col_size; i++)
            {
                if (done[i]) continue;

                done[i] = true;
                prev_k = i;
                k = permutation[i];
                while (i != k)
                {
                    std::swap(A->idx2[prev_k + start], A->idx2[k + start]);
                    std::swap(vals[prev_k + start], vals[k + start]);
                    done[k] = true;
                    prev_k = k;
                    k = permutation[k];
                }
            }
        }
        else
        {
            for (int i = 0; i < col_size; i++)
            {
                if (done[i]) continue;

                done[i] = true;
                prev_k = i;
                k = permutation[i];
                while (i != k)
                {
                    std::swap(A->idx2[prev_k + start], A->idx2[k + start]);
                    done[k] = true;
                    prev_k = k;
                    k = permutation[k];
                }
            }
        }
    }

    A->sorted = true;
    A->diag_first = false;
}

void COOMatrix::sort()
{
    sort_helper(this, vals);
}
void BCOOMatrix::sort()
{
    sort_helper(this, vals);
}
void CSRMatrix::sort()
{
    sort_helper(this, vals);
}
void BSRMatrix::sort()
{
    sort_helper(this, vals);
}
void CSCMatrix::sort()
{
    sort_helper(this, vals);
}
void BSCMatrix::sort()
{
    sort_helper(this, vals);
}


/**************************************************************
*****   Matrix Move Diagonal
**************************************************************
***** Moves the diagonal element to the front of each row
***** If matrix is not sorted, sorts before moving
**************************************************************/
template <typename T>
void move_diag_helper(COOMatrix* A, aligned_vector<T>& vals)
{
    if (A->diag_first || A->nnz == 0)
    {
        return;
    }

    if (!A->sorted)
    {
        A->sort();
    }

    int row_start, prev_row;
    int row, col;

    // Move diagonal entry to first in row
    row_start = 0;
    prev_row = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        row = A->idx1[i];
        col = A->idx2[i];
        if (row != prev_row)
        {
            prev_row = row;
            row_start = i;
        }
        else if (row == col)
        {
            auto tmp = vals[i];
            for (int j = i; j > row_start; j--)
            {
                A->idx2[j] = A->idx2[j-1];
                vals[j] = vals[j-1];
            }
            A->idx2[row_start] = row;
            vals[row_start] = tmp;
        }
    }

    A->diag_first = true;
}

template <typename T>
void move_diag_helper(CSRMatrix* A, aligned_vector<T>& vals)
{
    int start, end;
    int col;

    if (A->diag_first || A->nnz == 0)
    {
        return;
    }

    // Move diagonal values to beginning of each row
    if (A->data_size())
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            start = A->idx1[i];
            end = A->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A->idx2[j];
                if (col == i)
                {
                    auto tmp = vals[j];
                    for (int k = j; k > start; k--)
                    {
                        A->idx2[k] = A->idx2[k-1];
                        vals[k] = vals[k-1];
                    }
                    A->idx2[start] = i;
                    vals[start] = tmp;
                    break;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            start = A->idx1[i];
            end = A->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = A->idx2[j];
                if (col == i)
                {
                    for (int k = j; k > start; k--)
                    {
                        A->idx2[k] = A->idx2[k-1];
                    }
                    A->idx2[start] = i;
                    break;
                }
            }
        }
    }
    A->diag_first = true;
}

template <typename T>
void move_diag_helper(CSCMatrix* A, aligned_vector<T>& vals)
{
    int start, end;
    int row;

    if (A->diag_first || A->nnz == 0)
    {
        return;
    }

    // Move diagonal values to beginning of each row
    if (A->data_size())
    {
        for (int i = 0; i < A->n_cols; i++)
        {
            start = A->idx1[i];
            end = A->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = A->idx2[j];
                if (row == i)
                {
                    auto tmp = vals[j];
                    for (int k = j; k > start; k--)
                    {
                        A->idx2[k] = A->idx2[k-1];
                        vals[k] = vals[k-1];
                    }
                    A->idx2[start] = i;
                    vals[start] = tmp;
                    break;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < A->n_cols; i++)
        {
            start = A->idx1[i];
            end = A->idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = A->idx2[j];
                if (row == i)
                {
                    for (int k = j; k > start; k--)
                    {
                        A->idx2[k] = A->idx2[k-1];
                    }
                    A->idx2[start] = i;
                    break;
                }
            }
        }
    }
    A->diag_first = true;
}

void COOMatrix::move_diag()
{
    move_diag_helper(this, vals);
}
void BCOOMatrix::move_diag()
{
    move_diag_helper(this, vals);
}
void CSRMatrix::move_diag()
{
    move_diag_helper(this, vals);
}
void BSRMatrix::move_diag()
{
    move_diag_helper(this, vals);
}
void CSCMatrix::move_diag()
{
    move_diag_helper(this, vals);
}
void BSCMatrix::move_diag()
{
    move_diag_helper(this, vals);
}

/**************************************************************
*****   Matrix Removes Duplicates
**************************************************************
***** Goes thorugh each sorted row, and removes duplicate
***** entries, summing associated values
**************************************************************/
template <typename T>
void remove_duplicates_helper(COOMatrix* A, aligned_vector<T>& vals)
{
    if (!A->sorted)
    {
        A->sort();
        A->diag_first = false;
    }

    int prev_row, prev_col, ctr;
    int row, col;

    // Remove duplicates (sum together)
    prev_row = A->idx1[0];
    prev_col = A->idx2[0];
    ctr = 1;
    for (int i = 1; i < A->nnz; i++)
    {
        row = A->idx1[i];
        col = A->idx2[i];
        if (row == prev_row && col == prev_col)
        {
            A->append_vals(&vals[ctr - 1], &vals[i]);
        }
        else
        { 
            if (ctr != i)
            {
                A->idx1[ctr] = row;
                A->idx2[ctr] = col;
                vals[ctr] = vals[i];
            }
            ctr++;

            prev_row = row;
            prev_col = col;
        }
    }

    A->nnz = ctr;
}

template <typename T>
void remove_duplicates_helper(CSRMatrix* A, aligned_vector<T>& vals)
{
    int orig_start, orig_end;
    int new_start;
    int col, prev_col;
    int ctr, row_size;

    if (!A->sorted)
    {
        A->sort();
        A->diag_first = false;
    }

    orig_start = A->idx1[0];
    for (int row = 0; row < A->n_rows; row++)
    {
        new_start = A->idx1[row];
        orig_end = A->idx1[row+1];
        row_size = orig_end - orig_start;
        if (row_size == 0) 
        {
            orig_start = orig_end;
            A->idx1[row+1] = A->idx1[row];
            continue;
        }

        // Remove Duplicates
        col = A->idx2[orig_start];
        A->idx2[new_start] = col;
        vals[new_start] = vals[orig_start];
        prev_col = col;
        ctr = 1;
        for (int j = orig_start + 1; j < orig_end; j++)
        {
            col = A->idx2[j];
            if (col == prev_col)
            {
                A->append_vals(&vals[ctr - 1 + new_start], &vals[j]);
            }
            else
            {
                if (A->abs_val(vals[ctr - 1 + new_start]) < zero_tol)
                {
                    ctr--;
                }

                A->idx2[ctr + new_start] = col;
                vals[ctr + new_start] = vals[j];
                ctr++;
                prev_col = col;
            }
        }
        if (A->abs_val(vals[ctr - 1 + new_start]) < zero_tol)
        {
            ctr--;
        }

        orig_start = orig_end;
        A->idx1[row+1] = A->idx1[row] + ctr;
    }
    A->nnz = A->idx1[A->n_rows];
    A->idx2.resize(A->nnz);
    vals.resize(A->nnz);
}

template <typename T>
void remove_duplicates_helper(CSCMatrix* A, aligned_vector<T>& vals)
{
    int orig_start, orig_end;
    int new_start;
    int row, prev_row;
    int ctr, col_size;

    if (!A->sorted)
    {
        A->sort();
        A->diag_first = false;
    }

    orig_start = A->idx1[0];
    for (int col = 0; col < A->n_cols; col++)
    {
        new_start = A->idx1[col];
        orig_end = A->idx1[col+1];
        col_size = orig_end - orig_start;
        if (col_size == 0) 
        {
            orig_start = orig_end;
            A->idx1[col+1] = A->idx1[col];
            continue;
        }

        // Remove Duplicates
        row = A->idx2[orig_start];
        A->idx2[new_start] = row;
        vals[new_start] = vals[orig_start];
        prev_row = row;
        ctr = 1;
        for (int j = orig_start + 1; j < orig_end; j++)
        {
            row = A->idx2[j];
            if (row == prev_row)
            {
                A->append_vals(&vals[ctr - 1 + new_start], &vals[j]);
            }
            else
            {
                if (A->abs_val(vals[ctr - 1 + new_start]) < zero_tol)
                {
                    ctr--;
                }

                A->idx2[ctr + new_start] = row;
                vals[ctr + new_start] = vals[j];
                ctr++;
                prev_row = row;
            }
        }
        if (A->abs_val(vals[ctr - 1 + new_start]) < zero_tol)
        {
            ctr--;
        }

        orig_start = orig_end;
        A->idx1[col+1] = A->idx1[col] + ctr;
    }
    A->nnz = A->idx1[A->n_cols];
    A->idx2.resize(A->nnz);
    vals.resize(A->nnz);
}

void COOMatrix::remove_duplicates()
{
    remove_duplicates_helper(this, vals);
}
void BCOOMatrix::remove_duplicates()
{
    remove_duplicates_helper(this, vals);
}
void CSRMatrix::remove_duplicates()
{
    remove_duplicates_helper(this, vals);
}
void BSRMatrix::remove_duplicates()
{
    remove_duplicates_helper(this, vals);
}
void CSCMatrix::remove_duplicates()
{
    remove_duplicates_helper(this, vals);
}
void BSCMatrix::remove_duplicates()
{
    remove_duplicates_helper(this, vals);
}

/**************************************************************
*****   Matrix Convert
**************************************************************
***** Convert from one type of matrix to another
***** No copies if matrix type remains the same
***** If blocked matrix, converts to block matrix
**************************************************************/
COOMatrix* COOMatrix::to_COO()
{
    return this;
}
COOMatrix* BCOOMatrix::to_COO()
{
    return this;
}

CSRMatrix* COOMatrix::to_CSR()
{
    CSRMatrix* A = new CSRMatrix();
    COO_to_CSR(this, A, vals, A->vals);
    return A;
}
CSRMatrix* BCOOMatrix::to_CSR()
{
    BSRMatrix* A = new BSRMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    COO_to_CSR(this, A, vals, A->vals);
    return A;
}

CSCMatrix* COOMatrix::to_CSC()
{
    CSCMatrix* A = new CSCMatrix();
    COO_to_CSC(this, A, vals, A->vals);
    return A;
}
CSCMatrix* BCOOMatrix::to_CSC()
{
    BSCMatrix* A = new BSCMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    COO_to_CSC(this, A, vals, A->vals);
    return A;
}

COOMatrix* CSRMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    CSR_to_COO(this, A, vals, A->vals);
    return A;
}
COOMatrix* BSRMatrix::to_COO()
{
    BCOOMatrix* A = new BCOOMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    CSR_to_COO(this, A, vals, A->vals);
    return A;
}

CSRMatrix* CSRMatrix::to_CSR()
{
    return this;
}
CSRMatrix* BSRMatrix::to_CSR()
{
    return this;
}

CSCMatrix* CSRMatrix::to_CSC()
{
    CSCMatrix* A = new CSCMatrix();
    CSR_to_CSC(this, A, vals, A->vals);
    return A;
}
CSCMatrix* BSRMatrix::to_CSC()
{
    BSCMatrix* A = new BSCMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    CSR_to_CSC(this, A, vals, A->vals);
    return A;
}

COOMatrix* CSCMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    CSC_to_COO(this, A, vals, A->vals);
    return A;
}
COOMatrix* BSCMatrix::to_COO()
{
    BCOOMatrix* A = new BCOOMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    CSC_to_COO(this, A, vals, A->vals);
    return A;
}

CSRMatrix* CSCMatrix::to_CSR()
{
    CSRMatrix* A = new CSRMatrix();
    CSC_to_CSR(this, A, vals, A->vals);
    return A;
}
CSRMatrix* BSCMatrix::to_CSR()
{
    BSRMatrix* A = new BSRMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    CSC_to_CSR(this, A, vals, A->vals);
    return A;
}
CSCMatrix* CSCMatrix::to_CSC()
{
   return this; 
}
CSCMatrix* BSCMatrix::to_CSC()
{
    return this;
}

COOMatrix* COOMatrix::copy()
{
    COOMatrix* A = new COOMatrix();
    COO_to_COO(this, A, vals, A->vals);
    return A;
}
BCOOMatrix* BCOOMatrix::copy()
{
    BCOOMatrix* A = new BCOOMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    COO_to_COO(this, A, vals, A->vals);
    return A;
}
CSRMatrix* CSRMatrix::copy()
{
    CSRMatrix* A = new CSRMatrix();
    CSR_to_CSR(this, A, vals, A->vals);
    return A;
}
BSRMatrix* BSRMatrix::copy()
{
    BSRMatrix* A = new BSRMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    CSR_to_CSR(this, A, vals, A->vals);
    return A;
}
CSCMatrix* CSCMatrix::copy()
{
    CSCMatrix* A = new CSCMatrix();
    CSC_to_CSC(this, A, vals, A->vals);
    return A;
}
BSCMatrix* BSCMatrix::copy()
{
    BSCMatrix* A = new BSCMatrix();
    A->b_rows = b_rows;
    A->b_cols = b_cols;
    A->b_size = b_size;
    CSC_to_CSC(this, A, vals, A->vals);
    return A;
}


