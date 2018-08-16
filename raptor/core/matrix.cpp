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
void COOMatrix::print()
{
    int row, col;
    double val;

    for (int i = 0; i < nnz; i++)
    {
        row = idx1[i];
        col = idx2[i];
        val_print(row, col, vals[i]);
    }
}

void CSRMatrix::print()
{
    int col, start, end;

    for (int row = 0; row < n_rows; row++)
    {
        start = idx1[row];
        end = idx1[row+1];
        for (int j = start; j < end; j++)
        {
            col = idx2[j];
            val_print(row, col, vals[j]);
        }
    }
}
void CSCMatrix::print()
{
    int row, start, end;

    for (int col = 0; col < n_cols; col++)
    {
        start = idx1[col];
        end = idx1[col+1];
        for (int j = start; j < end; j++)
        {
            row = idx2[j];
            val_print(row, col, vals[j]);
        }
    }
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
void COOMatrix::copy_helper(const COOMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.clear();
    idx2.clear();
    vals.clear();

    idx1.reserve(A->nnz);
    idx2.reserve(A->nnz);
    vals.reserve(A->nnz);
    for (int i = 0; i < A->nnz; i++)
    {
        idx1.push_back(A->idx1[i]);
        idx2.push_back(A->idx2[i]);
        vals.push_back(copy_val(A->vals[i]));
    }
}
void BCOOMatrix::copy_helper(const BCOOMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    COOMatrix::copy_helper(A);             
} 

void COOMatrix::copy_helper(const CSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.clear();
    idx2.clear();
    vals.clear();

    idx1.reserve(A->nnz);
    idx2.reserve(A->nnz);
    vals.reserve(A->nnz);
    for (int i = 0; i < A->n_rows; i++)
    {
        int row_start = A->idx1[i];
        int row_end = A->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx1.push_back(i);
            idx2.push_back(A->idx2[j]);
            vals.push_back(copy_val(A->vals[j]));
        }
    }
}
void BCOOMatrix::copy_helper(const BSRMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    COOMatrix::copy_helper(A);             
} 
void COOMatrix::copy_helper(const CSCMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.clear();
    idx2.clear();
    vals.clear();

    idx1.reserve(A->nnz);
    idx2.reserve(A->nnz);
    vals.reserve(A->nnz);
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            idx1.push_back(A->idx2[j]);
            idx2.push_back(i);
            vals.push_back(copy_val(A->vals[j]));
        }
    }
}
void BCOOMatrix::copy_helper(const BSCMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    COOMatrix::copy_helper(A);             
} 

void CSRMatrix::copy_helper(const COOMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.resize(n_rows + 1);
    std::fill(idx1.begin(), idx1.end(), 0);
    if (nnz)
    {
        idx2.resize(nnz);
        if (A->vals.size())
            vals.resize(nnz);
    }

    // Calculate indptr
    for (int i = 0; i < nnz; i++)
    {
        int row = A->idx1[i];
        idx1[row+1]++;
    }
    for (int i = 0; i < n_rows; i++)
    {
        idx1[i+1] += idx1[i];
    }

    // Add indices and data
    aligned_vector<int> ctr;
    if (n_rows)
    {
            ctr.resize(n_rows, 0);
    }
    for (int i = 0; i < nnz; i++)
    {
        int row = A->idx1[i];
        int col = A->idx2[i];
        int index = idx1[row] + ctr[row]++;
        idx2[index] = col;
        if (A->vals.size()) // Checking that matrix has values (not S)
        {
            vals[index] = copy_val(A->vals[i]);
        }
    }
}
void BSRMatrix::copy_helper(const BCOOMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    CSRMatrix::copy_helper(A);
}

void CSRMatrix::copy_helper(const CSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.resize(A->n_rows + 1);
    idx2.resize(A->nnz);
    vals.resize(A->nnz);

    idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        idx1[i+1] = A->idx1[i+1];
        int row_start = idx1[i];
        int row_end = idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            idx2[j] = A->idx2[j];
            vals[j] = copy_val(A->vals[j]);
        }
    }
}
void BSRMatrix::copy_helper(const BSRMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    CSRMatrix::copy_helper(A);    
} 

void CSRMatrix::copy_helper(const CSCMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.clear();
    idx2.clear();
    vals.clear();

    // Resize vectors to appropriate dimensions
    idx1.resize(A->n_rows + 1);
    idx2.resize(A->nnz);
    if (A->vals.size())
        vals.resize(A->nnz);

    // Create indptr, summing number times row appears in CSC
    for (int i = 0; i <= A->n_rows; i++) idx1[i] = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        idx1[A->idx2[i] + 1]++;
    }
    for (int i = 1; i <= A->n_rows; i++)
    {
        idx1[i] += idx1[i-1];
    }

    // Add values to indices and data
    aligned_vector<int> ctr(n_rows, 0);
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row = A->idx2[j];
            int idx = idx1[row] + ctr[row]++;
            idx2[idx] = i;
            if (A->vals.size())
            {
                vals[idx] = copy_val(A->vals[j]);
            }
        }
    }
}
void BSRMatrix::copy_helper(const BSCMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    CSRMatrix::copy_helper(A);
}


void CSCMatrix::copy_helper(const COOMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.resize(n_cols + 1);
    std::fill(idx1.begin(), idx1.end(), 0);
    if (nnz)
    {
        idx2.resize(nnz);
        if (A->vals.size())
            vals.resize(nnz);
    }

    // Calculate indptr
    for (int i = 0; i < nnz; i++)
    {
        int col = A->idx1[i];
        idx1[col+1]++;
    }
    for (int i = 0; i < n_cols; i++)
    {
        idx1[i+1] += idx1[i];
    }

    // Add indices and data
    aligned_vector<int> ctr;
    if (n_cols)
    {
        ctr.resize(n_cols, 0);
    }
    for (int i = 0; i < nnz; i++)
    {
        int col = A->idx1[i];
        int row = A->idx2[i];
        int index = idx1[col] + ctr[col]++;
        idx2[index] = row;
        if (A->vals.size()) // Checking that matrix has values (not S)
        {
            vals[index] = copy_val(A->vals[i]);
        }
    }
}
void BSCMatrix::copy_helper(const BCOOMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    CSCMatrix::copy_helper(A);
}

void CSCMatrix::copy_helper(const CSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.clear();
    idx2.clear();
    vals.clear();

    // Resize vectors to appropriate dimensions
    idx1.resize(A->n_cols + 1);
    idx2.resize(A->nnz);
    if (A->vals.size())
        vals.resize(A->nnz);

    // Create indptr, summing number times row appears in CSC
    for (int i = 0; i <= A->n_cols; i++) idx1[i] = 0;
    for (int i = 0; i < A->nnz; i++)
    {
        idx1[A->idx2[i] + 1]++;
    }
    for (int i = 1; i <= A->n_cols; i++)
    {
        idx1[i] += idx1[i-1];
    }

    // Add values to indices and data
    aligned_vector<int> ctr(n_cols, 0);
    for (int i = 0; i < A->n_rows; i++)
    {
        int row_start = A->idx1[i];
        int row_end = A->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = A->idx2[j];
            int idx = idx1[col] + ctr[col]++;
            idx2[idx] = i;
            if (A->vals.size())
            {
                vals[idx] = copy_val(A->vals[j]);
            }
        }
    }
}

void BSCMatrix::copy_helper(const BSRMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    CSCMatrix::copy_helper(A);
}

void CSCMatrix::copy_helper(const CSCMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.resize(A->n_cols + 1);
    idx2.resize(A->nnz);
    vals.resize(A->nnz);

    idx1[0] = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        idx1[i+1] = col_end;
        for (int j = col_start; j < col_end; j++)
        {
            idx2[j] = A->idx2[j];
            vals[j] = A->vals[j];
        }
    }
}
void BSCMatrix::copy_helper(const BSCMatrix* A)
{
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;

    CSCMatrix::copy_helper(A);

    for (int i = 0; i < A->nnz; i++)
    {
        double* new_ptr = new double[b_size];
        double* val_ptr = vals[i];
        for (int j = 0; j < b_size; j++)
        {
            new_ptr[j] = val_ptr[j];
        }
        vals[i] = new_ptr;
    }                
} 



/**************************************************************
*****   Matrix Sort
**************************************************************
***** Sorts the sparse matrix by row and column
**************************************************************/
void COOMatrix::sort()
{
    if (sorted || nnz == 0)
    {
        sorted = true;
        return;
    }

    int k, prev_k;

    aligned_vector<int> permutation(nnz);
    aligned_vector<bool> done(nnz, false);

    // Create permutation vector p
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(),
        [&](int i, int j){ 
            if (idx1[i] == idx1[j])
                return idx2[i] < idx2[j];
            else
                return idx1[i] < idx1[j];
        });

    // Permute all vectors (rows, cols, data) 
    // according to p
    for (int i = 0; i < nnz; i++)
    {
        if (done[i]) continue;

        done[i] = true;
        prev_k = i;
        k = permutation[i];
        while (i != k)
        {
            std::swap(idx1[prev_k], idx1[k]);
            std::swap(idx2[prev_k], idx2[k]);
            std::swap(vals[prev_k], vals[k]);
            done[k] = true;
            prev_k = k;
            k = permutation[k];
        }
    }

    sorted = true;
    diag_first = false;
}

void CSRMatrix::sort()
{
    int start, end, row_size;
    int k, prev_k;

    if (sorted || nnz == 0)
    {
        sorted = true;
        return;
    }

    aligned_vector<int> permutation;
    aligned_vector<bool> done;

    // Sort the columns of each row (and data accordingly) and remove
    // duplicates (summing values together)
    for (int row = 0; row < n_rows; row++)
    {
        start = idx1[row];
        end = idx1[row+1];
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
                    return idx2[i+start] < idx2[j+start];
                });


        // Permute columns and data according to p
        done.resize(row_size);
        for (int i = 0; i < row_size; i++)
        {
            done[i] = false;
        }
        if (vals.size())
        {
            for (int i = 0; i < row_size; i++)
            {
                if (done[i]) continue;

                done[i] = true;
                prev_k = i;
                k = permutation[i];
                while (i != k)
                {
                    std::swap(idx2[prev_k + start], idx2[k + start]);
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
                    std::swap(idx2[prev_k + start], idx2[k + start]);
                    done[k] = true;
                    prev_k = k;
                    k = permutation[k];
                }
            }
        }
    }

    sorted = true;
    diag_first = false;
}

void CSCMatrix::sort()
{
    int start, end, col_size;
    int k, prev_k;

    if (sorted || nnz == 0)
    {
        sorted = true;
        return;
    }

    aligned_vector<int> permutation;
    aligned_vector<bool> done;

    // Sort the columns of each col (and data accordingly) and remove
    // duplicates (summing values together)
    for (int col = 0; col < n_cols; col++)
    {
        start = idx1[col];
        end = idx1[col+1];
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
                    return idx2[i+start] < idx2[j+start];
                });


        // Permute columns and data according to p
        done.resize(col_size);
        for (int i = 0; i < col_size; i++)
        {
            done[i] = false;
        }
        if (vals.size())
        {
            for (int i = 0; i < col_size; i++)
            {
                if (done[i]) continue;

                done[i] = true;
                prev_k = i;
                k = permutation[i];
                while (i != k)
                {
                    std::swap(idx2[prev_k + start], idx2[k + start]);
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
                    std::swap(idx2[prev_k + start], idx2[k + start]);
                    done[k] = true;
                    prev_k = k;
                    k = permutation[k];
                }
            }
        }
    }

    sorted = true;
    diag_first = false;
}



/**************************************************************
*****   Matrix Move Diagonal
**************************************************************
***** Moves the diagonal element to the front of each row
***** If matrix is not sorted, sorts before moving
**************************************************************/
void COOMatrix::move_diag()
{
    if (diag_first || nnz == 0)
    {
        return;
    }

    if (!sorted)
    {
        sort();
    }

    int row_start, prev_row;
    int row, col;

    // Move diagonal entry to first in row
    row_start = 0;
    prev_row = 0;
    for (int i = 0; i < nnz; i++)
    {
        row = idx1[i];
        col = idx2[i];
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
                idx2[j] = idx2[j-1];
                vals[j] = vals[j-1];
            }
            idx2[row_start] = row;
            vals[row_start] = tmp;
        }
    }

    diag_first = true;
}

void CSRMatrix::move_diag()
{
    int start, end;
    int col;

    if (diag_first || nnz == 0)
    {
        return;
    }

    // Move diagonal values to beginning of each row
    if (vals.size())
    {
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = idx2[j];
                if (col == i)
                {
                    auto tmp = vals[j];
                    for (int k = j; k > start; k--)
                    {
                        idx2[k] = idx2[k-1];
                        vals[k] = vals[k-1];
                    }
                    idx2[start] = i;
                    vals[start] = tmp;
                    break;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < n_rows; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                col = idx2[j];
                if (col == i)
                {
                    for (int k = j; k > start; k--)
                    {
                        idx2[k] = idx2[k-1];
                    }
                    idx2[start] = i;
                    break;
                }
            }
        }
    }
    diag_first = true;
}

void CSCMatrix::move_diag()
{
    int start, end;
    int row;

    if (diag_first || nnz == 0)
    {
        return;
    }

    // Move diagonal values to beginning of each row
    if (vals.size())
    {
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = idx2[j];
                if (row == i)
                {
                    auto tmp = vals[j];
                    for (int k = j; k > start; k--)
                    {
                        idx2[k] = idx2[k-1];
                        vals[k] = vals[k-1];
                    }
                    idx2[start] = i;
                    vals[start] = tmp;
                    break;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < n_cols; i++)
        {
            start = idx1[i];
            end = idx1[i+1];
            for (int j = start; j < end; j++)
            {
                row = idx2[j];
                if (row == i)
                {
                    for (int k = j; k > start; k--)
                    {
                        idx2[k] = idx2[k-1];
                    }
                    idx2[start] = i;
                    break;
                }
            }
        }
    }
    diag_first = true;
}



/**************************************************************
*****   Matrix Removes Duplicates
**************************************************************
***** Goes thorugh each sorted row, and removes duplicate
***** entries, summing associated values
**************************************************************/
void COOMatrix::remove_duplicates()
{
    if (!sorted)
    {
        sort();
        diag_first = false;
    }

    int prev_row, prev_col, ctr;
    int row, col;

    // Remove duplicates (sum together)
    prev_row = idx1[0];
    prev_col = idx2[0];
    ctr = 1;
    for (int i = 1; i < nnz; i++)
    {
        row = idx1[i];
        col = idx2[i];
        if (row == prev_row && col == prev_col)
        {
            append_vals(&vals[ctr - 1], vals[i]);
        }
        else
        { 
            if (ctr != i)
            {
                idx1[ctr] = row;
                idx2[ctr] = col;
                vals[ctr] = vals[i];
            }
            ctr++;

            prev_row = row;
            prev_col = col;
        }
    }

    nnz = ctr;
}

void CSRMatrix::remove_duplicates()
{
    int orig_start, orig_end;
    int new_start;
    int col, prev_col;
    int ctr, row_size;

    if (!sorted)
    {
        sort();
        diag_first = false;
    }

    orig_start = idx1[0];
    for (int row = 0; row < n_rows; row++)
    {
        new_start = idx1[row];
        orig_end = idx1[row+1];
        row_size = orig_end - orig_start;
        if (row_size == 0) 
        {
            orig_start = orig_end;
            idx1[row+1] = idx1[row];
            continue;
        }

        // Remove Duplicates
        col = idx2[orig_start];
        idx2[new_start] = col;
        vals[new_start] = vals[orig_start];
        prev_col = col;
        ctr = 1;
        for (int j = orig_start + 1; j < orig_end; j++)
        {
            col = idx2[j];
            if (col == prev_col)
            {
                append_vals(&vals[ctr - 1 + new_start], vals[j]);
            }
            else
            {
                if (abs_val(vals[ctr - 1 + new_start]) < zero_tol)
                {
                    ctr--;
                }

                idx2[ctr + new_start] = col;
                vals[ctr + new_start] = vals[j];
                ctr++;
                prev_col = col;
            }
        }
        if (abs_val(vals[ctr - 1 + new_start]) < zero_tol)
        {
            ctr--;
        }

        orig_start = orig_end;
        idx1[row+1] = idx1[row] + ctr;
    }
    nnz = idx1[n_rows];
    idx2.resize(nnz);
    vals.resize(nnz);
}

void CSCMatrix::remove_duplicates()
{
    int orig_start, orig_end;
    int new_start;
    int row, prev_row;
    int ctr, col_size;

    if (!sorted)
    {
        sort();
        diag_first = false;
    }

    orig_start = idx1[0];
    for (int col = 0; col < n_cols; col++)
    {
        new_start = idx1[col];
        orig_end = idx1[col+1];
        col_size = orig_end - orig_start;
        if (col_size == 0) 
        {
            orig_start = orig_end;
            idx1[col+1] = idx1[col];
            continue;
        }

        // Remove Duplicates
        row = idx2[orig_start];
        idx2[new_start] = row;
        vals[new_start] = vals[orig_start];
        prev_row = row;
        ctr = 1;
        for (int j = orig_start + 1; j < orig_end; j++)
        {
            row = idx2[j];
            if (row == prev_row)
            {
                append_vals(&vals[ctr - 1 + new_start], vals[j]);
            }
            else
            {
                if (abs_val(vals[ctr - 1 + new_start]) < zero_tol)
                {
                    ctr--;
                }

                idx2[ctr + new_start] = row;
                vals[ctr + new_start] = vals[j];
                ctr++;
                prev_row = row;
            }
        }
        if (abs_val(vals[ctr - 1 + new_start]) < zero_tol)
        {
            ctr--;
        }

        orig_start = orig_end;
        idx1[row+1] = idx1[row] + ctr;
    }
    nnz = idx1[n_cols];
    idx2.resize(nnz);
    vals.resize(nnz);
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
    A->copy_helper(this);
    return A;
}
CSRMatrix* BCOOMatrix::to_CSR()
{
    BSRMatrix* A = new BSRMatrix();
    A->copy_helper(this);
    return A;
}

CSCMatrix* COOMatrix::to_CSC()
{
    CSCMatrix* A = new CSCMatrix();
    A->copy_helper(this);
    return A;
}
CSCMatrix* BCOOMatrix::to_CSC()
{
    BSCMatrix* A = new BSCMatrix();
    A->copy_helper(this);
    return A;
}

COOMatrix* CSRMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    A->copy_helper(this);
    return A;
}
COOMatrix* BSRMatrix::to_COO()
{
    BCOOMatrix* A = new BCOOMatrix();
    A->copy_helper(this);
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
    A->copy_helper(this);
    return A;
}
CSCMatrix* BSRMatrix::to_CSC()
{
    BSCMatrix* A = new BSCMatrix();
    A->copy_helper(this);
    return A;
}

COOMatrix* CSCMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    A->copy_helper(this);
    return A;
}
COOMatrix* BSCMatrix::to_COO()
{
    BCOOMatrix* A = new BCOOMatrix();
    A->copy_helper(this);
    return A;
}

CSRMatrix* CSCMatrix::to_CSR()
{
    CSRMatrix* A = new CSRMatrix();
    A->copy_helper(this);
    return A;
}
CSRMatrix* BSCMatrix::to_CSR()
{
    BSRMatrix* A = new BSRMatrix();
    A->copy_helper(this);
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


