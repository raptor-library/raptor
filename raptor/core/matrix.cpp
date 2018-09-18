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
        val = vals[i];

        printf("A[%d][%d] = %e\n", row, col, val);
    }
}
void CSRMatrix::print()
{
    int col, start, end;
    double val;

    for (int row = 0; row < n_rows; row++)
    {
        start = idx1[row];
        end = idx1[row+1];
        for (int j = start; j < end; j++)
        {
            col = idx2[j];
            val = vals[j];

            printf("A[%d][%d] = %e\n", row, col, val);
        }
    }
}
void CSCMatrix::print()
{
    int row, start, end;
    double val;

    for (int col = 0; col < n_cols; col++)
    {
        start = idx1[col];
        end = idx1[col+1];
        for (int j = start; j < end; j++)
        {
            row = idx2[j];
            val = vals[j];

            printf("A[%d][%d] = %e\n", row, col, val);
        }
    }
}

void BSRMatrix::print()
{
    int col, start, end;

    for (int i = 0; i < n_rows/b_rows; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            // Call block print function
	    block_print(i, j, idx2[j]);
	    printf("----------\n");
        }
    }
}

void BSRMatrix::block_print(int row, int num_blocks_prev, int col)
{
    // The upper left corner indices of the matrix
    int upper_i = row * b_rows;
    int upper_j = col * b_cols;
    // The offset to determine where this block starts in the data array
    int data_offset = num_blocks_prev * b_size;

    int glob_i, glob_j, ind;
    double val;
    for (int i=0; i<b_rows; i++)
    {
        for (int j=0; j<b_cols; j++)
	{
            glob_i = upper_i + i;
	    glob_j = upper_j + j;
	    ind = i * b_cols + j + data_offset;
	    val = vals[ind];
	    printf("A[%d][%d] = %e\n", glob_i, glob_j, val);
	}
    }
}


Matrix* COOMatrix::transpose()
{
    Matrix* T = new COOMatrix(n_rows, n_cols, idx2, idx1, vals);

    return T;
}

Matrix* CSRMatrix::transpose()
{
    // Create CSC Matrix... rowptr is now colptr
    CSCMatrix* T_csc = new CSCMatrix(n_rows, n_cols, idx1, idx2, vals); 

    // Convert back to CSR to tranpose
    Matrix* T = T_csc->to_CSR();

    delete T_csc;

    return T;
}

Matrix* CSCMatrix::transpose()
{
    // Create CSR Matrix... colptr is now rowptr
    CSRMatrix* T_csr = new CSRMatrix(n_rows, n_cols, idx1, idx2, vals); 

    // Convert back to CSC to tranpose
    Matrix* T = T_csr->to_CSC();

    delete T_csr;

    return T;
}

Matrix* BSRMatrix::transpose()
{
    printf("Currently not implemented.\n");	
    return NULL;
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
*****  COOMatrix Add Value
**************************************************************
***** Inserts value into the position (row, col) of the matrix
*****
***** Parameters
***** -------------
***** row : int
*****    Row in which to insert value 
***** col : int
*****    Column in which to insert value
***** value : double
*****    Nonzero value to be inserted into the matrix
**************************************************************/
void COOMatrix::add_value(int row, int col, double value)
{
    idx1.push_back(row);
    idx2.push_back(col);
    vals.push_back(value);
    nnz++;
}

void COOMatrix::add_block(int row, int col, aligned_vector<double>& values){
    printf("Not implemented.\n");
}

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
        vals.push_back(A->vals[i]);
    }
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
            vals.push_back(A->vals[j]);
        }
    }
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
            vals.push_back(A->vals[j]);
        }
    }
}

void COOMatrix::copy_helper(const BSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = 0;

    idx1.clear();
    idx2.clear();
    vals.clear();

    idx1.reserve(A->nnz);
    idx2.reserve(A->nnz);
    vals.reserve(A->nnz);

    for (int i = 0; i < n_rows/A->b_rows; i++)
    {
        int row_start = A->idx1[i];
        int row_end = A->idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            // Call block copy function
	    block_copy_helper(A, i, j, A->idx2[j]);
        }
    }
}

void COOMatrix::block_copy_helper(const BSRMatrix* A, int row, int num_blocks_prev, int col)
{
    int upper_i = row * A->b_rows;
    int upper_j = col * A->b_cols;
    int data_offset = num_blocks_prev * A->b_size;

    int glob_i, glob_j, ind;
    double val;
    for (int i = 0; i < A->b_rows; i++)
    {
        for (int j = 0; j < A->b_cols; j++)
	{
            glob_i = upper_i + i;
	    glob_j = upper_j + j;
	    ind = i * A->b_cols + j + data_offset;
	    val = A->vals[ind];

	    if (fabs(val) > zero_tol)
	    {
	        idx1.push_back(glob_i);
	        idx2.push_back(glob_j);
                vals.push_back(val);
		nnz++;
	    }
	}
    }
}

/**************************************************************
*****   COOMatrix to_dense
**************************************************************
***** Converts the COOMatrix into a dense matrix
***** in the form of a flattened vector ordered row-wise
**************************************************************/
aligned_vector<double> COOMatrix::to_dense() const
{
    aligned_vector<double> dense(n_rows * n_cols);
    std::fill(dense.begin(), dense.end(), 0.0);

    for (int i = 0; i < nnz; i++)
    {
        dense[idx1[i]*n_cols + idx2[i]] = vals[i];
    }

    return dense;
}

/**************************************************************
*****   COOMatrix Sort
**************************************************************
***** Sorts the sparse matrix by row, and by column within 
***** each row.  Removes duplicates, summing their values 
***** together.
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
    double tmp;

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
            tmp = vals[i];
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

void COOMatrix::remove_duplicates()
{
    if (!sorted)
    {
        sort();
        diag_first = false;
    }

    int prev_row, prev_col, ctr;
    int row, col;
    double val;

    // Remove duplicates (sum together)
    prev_row = idx1[0];
    prev_col = idx2[0];
    ctr = 1;
    for (int i = 1; i < nnz; i++)
    {
        row = idx1[i];
        col = idx2[i];
        val = vals[i];
        if (row == prev_row && col == prev_col)
        {
            vals[ctr-1] += val;
        }
        else
        { 
            if (ctr != i)
            {
                idx1[ctr] = row;
                idx2[ctr] = col;
                vals[ctr] = val;
            }
            ctr++;

            prev_row = row;
            prev_col = col;
        }
    }

    nnz = ctr;
}

/**************************************************************
*****  CSRMatrix Add Value
**************************************************************
***** Inserts value into the position (row, col) of the matrix.
***** Values must be inserted in row-wise order, so if the row
***** is not equal to the row of the previously inserted value,
***** indptr is edited, and it is assumed that row is complete.
***** TODO -- this method needs further testing
*****
***** Parameters
***** -------------
***** row : int
*****    Row in which to insert value 
***** col : int
*****    Column in which to insert value
***** value : double
*****    Nonzero value to be inserted into the matrix
**************************************************************/
void CSRMatrix::add_value(int row, int col, double value)
{
    // Assumes idx1 is created separately
    idx2.push_back(col);
    vals.push_back(value);
    nnz++;
}

void CSRMatrix::add_block(int row, int col, aligned_vector<double>& values){
    printf("Not implemented.\n");
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
        if (A->vals.size())
        {
            double val = A->vals[i];
            vals[index] = val;
        }
    }
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
            vals[j] = A->vals[j];
        }
    }
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
                vals[idx] = A->vals[j];
            }
        }
    }
}
void CSRMatrix::copy_helper(const BSRMatrix* A)
{
    printf("Currently not implemented\n");
}

/**************************************************************
*****   CSRMatrix to_dense
**************************************************************
***** Converts the CSRMatrix into a dense matrix
***** in the form of a flattened vector ordered row-wise
**************************************************************/
aligned_vector<double> CSRMatrix::to_dense() const
{
    aligned_vector<double> dense(n_rows * n_cols);
    std::fill(dense.begin(), dense.end(), 0.0);

    for (int i = 0; i < n_rows; i++)
    {
        for (int j = idx1[i]; j < idx1[i+1]; j++)
	{
            dense[i*n_cols + idx2[j]] = vals[j];
	}
    }
    return dense;
}

/**************************************************************
*****   CSRMatrix Sort
**************************************************************
***** Sorts the sparse matrix by columns within each row.  
***** Removes duplicates, summing their values 
***** together.
**************************************************************/
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

void CSRMatrix::move_diag()
{
    int start, end;
    int col;
    double tmp;

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
                    tmp = vals[j];
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

void CSRMatrix::remove_duplicates()
{
    int orig_start, orig_end;
    int new_start;
    int col, prev_col;
    int ctr, row_size;
    double val;

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
        val = vals[orig_start];
        idx2[new_start] = col;
        vals[new_start] = val;
        prev_col = col;
        ctr = 1;
        for (int j = orig_start + 1; j < orig_end; j++)
        {
            col = idx2[j];
            val = vals[j];
            if (col == prev_col)
            {
                vals[ctr - 1 + new_start] += val;
            }
            else
            {
                if (fabs(vals[ctr - 1 + new_start]) < zero_tol)
                {
                    ctr--;
                }

                idx2[ctr + new_start] = col;
                vals[ctr + new_start] = val;
                ctr++;
                prev_col = col;
            }
        }
        if (fabs(vals[ctr - 1 + new_start]) < zero_tol)
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

/**************************************************************
*****  BSRMatrix Copy
**************************************************************
**************************************************************/
void BSRMatrix::copy_helper(const COOMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;

    aligned_vector<double> A_dense = A->to_dense();


    double block_dense[n_rows*n_cols];

    int block_ind, glob_i, glob_j;
    for (int k = 0; k < n_rows/b_rows; k++)
    {
        for (int w = 0; w < n_cols/b_cols; w++)
	{
            for (int i = 0; i < b_rows; i++)
	    {
                for (int j = 0; j < b_cols; j++)
		{
                    block_ind = k*n_cols/b_cols + w;
		    glob_i = k * b_rows + i;
		    glob_j = w * b_cols + j;
		    block_dense[block_ind * b_size + i * b_cols + j] = A_dense[glob_i * n_cols + glob_j];
		}
	    }
	}
    }    

    const BSRMatrix* B = new BSRMatrix(n_rows, n_cols, b_rows, b_cols, block_dense);
    copy_helper(B);

    delete B;
}

void BSRMatrix::copy_helper(const CSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;

    aligned_vector<double> A_dense = A->to_dense();
    double block_dense[n_rows*n_cols];

    int block_ind, glob_i, glob_j;
    for (int k = 0; k < n_rows/b_rows; k++)
    {
        for (int w = 0; w < n_cols/b_cols; w++)
        {
            for (int i = 0; i < b_rows; i++)
            {
                for (int j = 0; j < b_cols; j++)
                {
                    block_ind = k*n_cols/b_cols + w;
                    glob_i = k * b_rows + i;
                    glob_j = w * b_cols + j;
                    block_dense[block_ind * b_size + i * b_cols + j] = A_dense[glob_i * n_cols + glob_j];
                }
            }
        }
    }

    const BSRMatrix* B = new BSRMatrix(n_rows, n_cols, b_rows, b_cols, block_dense);
    copy_helper(B);

    delete B;
}

void BSRMatrix::copy_helper(const CSCMatrix* A)
{
    printf("Currently not implemented\n");
}

void BSRMatrix::copy_helper(const BSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    b_rows = A->b_rows;
    b_cols = A->b_cols;
    b_size = A->b_size;
    n_blocks = A->n_blocks;
    nnz = A->nnz;

    idx1.resize(A->idx1.size());
    idx2.resize(A->idx2.size());
    vals.resize(A->vals.size());

    std::copy(A->idx1.begin(), A->idx1.end(), idx1.begin());
    std::copy(A->idx2.begin(), A->idx2.end(), idx2.begin());
    std::copy(A->vals.begin(), A->vals.end(), vals.begin());
}

/**************************************************************
*****   BSRMatrix to_dense
**************************************************************
***** Converts the BSRMatrix into a dense matrix
***** in the form of a flattened vector ordered row-wise
**************************************************************/
aligned_vector<double> BSRMatrix::to_dense()
{
    aligned_vector<double> dense(n_rows * n_cols);
    std::fill(dense.begin(), dense.end(), 0.0);

    int start, end;
    int upper_i, upper_j, data_offset;
    double val;
    int glob_i, glob_j, ind;
    for (int i=0; i<n_rows/b_rows; i++)
    {
	start = idx1[i];
	end = idx1[i+1];
        for (int j=start; j<end; j++)
	{
            upper_i = i * b_rows;
	    upper_j = idx2[j] * b_cols;
	    data_offset = j * b_size;
	    for (int block_i = 0; block_i < b_rows; block_i++)
	    {
                for (int block_j = 0; block_j < b_cols; block_j++)
		{
                    glob_i = upper_i + block_i;
		    glob_j = upper_j + block_j;
		    ind = block_i * b_cols + block_j + data_offset;
		    val = vals[ind];
                    dense[glob_i*n_cols + glob_j] = val;
		}
	    }
	}
    }

    return dense;
}

/**************************************************************
*****  BSRMatrix Add Value
**************************************************************
**************************************************************/
void BSRMatrix::add_value(int row, int col, double value)
{
    printf("Currently not implemented\n");
}

/**************************************************************
*****  BSRMatrix Add Block
***************************************************************
***** Inserts nonzero block of values into position (row, col)
***** of the block-matrix - NOT the global indices.
***** Values of the non-zero block being added should be row
***** ordered within the values array.
*****
***** Input:
*****   row:
*****     row index of non-zero block in matrix - NOT global row
*****   col:
*****     col index of non-zero block in matrix - NOT global col
*****   values:
*****     vector of block values - ordered row-wise within the block
**************************************************************/
void BSRMatrix::add_block(int row, int col, aligned_vector<double>& values)
{
    // Only add correct number of elements for block if values is longer than
    // block size
    if (values.size() > b_size) values.erase(values.begin()+b_size, values.end());

    // Add zeros to end of values vector if smaller than block size
    if (values.size() < b_size)
    {
        for(int k=values.size(); k<b_size; k++)
	{
            values.push_back(0.0);
	}
    }

    int start, end, j, data_offset;
    start = idx1[row];
    end = idx1[row+1];
    data_offset = idx1[row] * b_size;

    // Update cols vector and data offset
    if(idx2.size() < 1)
    {
        // First block added
        idx2.push_back(col);
	data_offset = 0;
    }
    else if(start == end || col > idx2[end-1])
    {
        idx2.insert(idx2.begin()+end, col);
	data_offset += b_size * (end-start);
    }
    else if(col < idx2[start]){
        idx2.insert(idx2.begin()+start, col);
    }
    else
    {
        while(j < end)
        {
            if(col < idx2[j])
            {
                idx2.insert(idx2.begin()+j, col);
		data_offset += b_size * (j-start);
            }
	    else j++;
        }
    }

    // Update rowptr
    for(int i=row+1; i<idx1.size(); i++){
        idx1[i]++;
    }

    // Update vals array
    vals.insert(vals.begin()+data_offset, values.begin(), values.end());

    // Update matrix variables
    nnz += b_size;
    n_blocks++;

}

/**************************************************************
*****   BSRMatrix Sort
**************************************************************
**************************************************************/
void BSRMatrix::sort()
{
    //printf("Currently not implemented\n");
    return;
}

void BSRMatrix::move_diag()
{
    printf("Currently not implemented\n");
}

void BSRMatrix::remove_duplicates()
{
    printf("Currently not implemented\n");
}

/**************************************************************
*****  CSCMatrix Add Value
**************************************************************
***** Inserts value into the position (row, col) of the matrix.
***** Values must be inserted in column-wise order, so if the col
***** is not equal to the col of the previously inserted value,
***** indptr is edited, and it is assumed that col is complete.
***** TODO -- this method needs further testing
*****
***** Parameters
***** -------------
***** row : int
*****    Row in which to insert value 
***** col : int
*****    Column in which to insert value
***** value : double
*****    Nonzero value to be inserted into the matrix
**************************************************************/
void CSCMatrix::add_value(int row, int col, double value)
{
    // Assumes idx1 is created separately
    idx2.push_back(row);
    vals.push_back(value);
    nnz++;
}

void CSCMatrix::add_block(int row, int col, aligned_vector<double>& values){
    printf("Not implemented.\n");
}

void CSCMatrix::copy_helper(const COOMatrix* A)
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
    for (int i = 0; i < n_cols + 1; i++)
    {
        idx1[i] = 0;
    }
    for (int i = 0; i < A->nnz; i++)
    {
        idx1[A->idx2[i]+1]++;
    }
    for (int i = 0; i < A->n_cols; i++)
    {
        idx1[i+1] += idx1[i];
    }

    // Add indices and data
    aligned_vector<int> ctr;
    if (n_cols)
    {
        ctr.resize(n_cols, 0);
    }
    for (int i = 0; i < A->nnz; i++)
    {
        int row = A->idx1[i];
        int col = A->idx2[i];
        int index = idx1[col] + ctr[col]++;
        idx2[index] = row;
        if (A->vals.size())
        {
            double val = A->vals[i];
            vals[index] = val;
        }
    }        
}
void CSCMatrix::copy_helper(const CSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    // Resize vectors to appropriate dimensions
    idx1.resize(A->n_cols + 1);
    if (A->nnz)
    {
        idx2.resize(A->nnz);
        if (A->vals.size())
            vals.resize(A->nnz);
    }

    // Create indptr, summing number times col appears in CSR
    for (int i = 0; i <= A->n_cols; i++) 
    {
        idx1[i] = 0;
    }
    for (int i = 0; i < A->nnz; i++)
    {
        idx1[A->idx2[i] + 1]++;
    }
    for (int i = 0; i < A->n_cols; i++)
    {
        idx1[i+1] += idx1[i];
    }

    // Add values to indices and data
    if (A->n_cols)
    {
        aligned_vector<int> ctr(A->n_cols, 0);
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
                    vals[idx] = A->vals[j];
                }
            }
        }
    }
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

void CSCMatrix::copy_helper(const BSRMatrix* A)
{
    printf("Currently not implemented");
    return;
}

/**************************************************************
*****   CSCMatrix Sort
**************************************************************
***** Sorts the sparse matrix by rows within each column.  
***** Removes duplicates, summing their values 
***** together.
**************************************************************/
void CSCMatrix::sort()
{
    int start, end, col_size;
    int prev_k, k;

    aligned_vector<int> permutation;
    aligned_vector<bool> done;

    if (sorted || nnz == 0)
    {
        return;
    }

    // Sort the columns of each row (and data accordingly) and remove
    // duplicates (summing values together)
    for (int col = 0; col < n_cols; col++)
    {
        start  = idx1[col];
        end = idx1[col+1];
        col_size = end - start;
        if (col_size == 0)
        {
            continue;
        }

        // Create permutation vector p for row
        permutation.resize(col_size);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(),
                [&](int i, int j)
                { 
                    return idx2[i + start] < idx2[j + start];
                });

        // Permute columns and data according to p
        done.resize(col_size);
        for (int i = 0; i < col_size; i++)
        {
            done[i] = false;
        }
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

    sorted = true;
    diag_first = false;
}

void CSCMatrix::remove_duplicates()
{
    if (!sorted)
    {
        sort();
        diag_first = false;
    }
    
    int orig_start, orig_end, new_start;
    int col_size;
    int row, prev_row, ctr;
    double val;

    // Sort the columns of each row (and data accordingly) and remove
    // duplicates (summing values together)
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
        val = vals[orig_start];
        idx2[new_start] = row;
        vals[new_start] = val;
        prev_row = row;
        ctr = 1;
        for (int j = orig_start + 1; j < orig_end; j++)
        {
            row = idx2[j];
            val = vals[j];
            if (row == prev_row)
            {
                vals[ctr - 1 + new_start] += val;
            }
            else
            {
                idx2[ctr + new_start] = row;
                vals[ctr + new_start] = val;
                ctr++;
                prev_row = row;
            }
        }
        orig_start = orig_end;
        idx1[col+1] = idx1[col] + ctr;
    }
}

void CSCMatrix::move_diag()
{
    if (diag_first || nnz == 0)
    {
       return;
    }

    int start, end, row;
    double tmp;

    // Move diagonal values to beginning of each column
    for (int i = 0; i < n_cols; i++)
    {
        start = idx1[i];
        end = idx1[i+1];
        for (int j = start; j < end; j++)
        {
            row = idx2[j];
            if (row == i)
            {
                tmp = vals[j];
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

    diag_first = true;
}


COOMatrix* COOMatrix::to_COO()
{
    return this;
}

CSRMatrix* COOMatrix::to_CSR()
{
    CSRMatrix* A = new CSRMatrix();
    A->copy_helper(this);
    return A;
}

CSCMatrix* COOMatrix::to_CSC()
{
    CSCMatrix* A = new CSCMatrix();
    A->copy_helper(this);
    return A;
}

COOMatrix* CSRMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    A->copy_helper(this);
    return A;
}

CSRMatrix* CSRMatrix::to_CSR()
{
    return this;
}

CSCMatrix* CSRMatrix::to_CSC()
{
    CSCMatrix* A = new CSCMatrix();
    A->copy_helper(this);
    return A;
}

COOMatrix* CSCMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    A->copy_helper(this);
    return A;
}

CSRMatrix* CSCMatrix::to_CSR()
{
    CSRMatrix* A = new CSRMatrix();
    A->copy_helper(this);
    return A;
}

CSCMatrix* CSCMatrix::to_CSC()
{
   return this; 
}


COOMatrix* BSRMatrix::to_COO()
{
    COOMatrix* A = new COOMatrix();
    A->copy_helper(this);
    return A;
}

CSRMatrix* BSRMatrix::to_CSR()
{
    return NULL;
}

CSCMatrix* BSRMatrix::to_CSC()
{
    return NULL;
}


