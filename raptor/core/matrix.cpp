// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/matrix.hpp"

using namespace raptor;

/**************************************************************
*****  Matrix Print
**************************************************************
***** Print the nonzeros in the matrix, as well as the row
***** and column according to each nonzero
**************************************************************/
void Matrix::print()
{
    apply_func([] (int row, int col, double val)
        {
            printf("A[%d][%d] = %e\n", row, col, val);
        });
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

void COOMatrix::copy(const COOMatrix* A)
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
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}
void COOMatrix::copy(const CSRMatrix* A)
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
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}
void COOMatrix::copy(const CSCMatrix* A)
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
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}


/**************************************************************
*****   COOMatrix Condense Rows
**************************************************************
***** Removes zero rows from the matrix, and initializes 
***** row_list, which points from new row index to original 
***** row index.
**************************************************************/
void COOMatrix::condense_rows()
{
    std::set<int> row_set;
    std::map<int, int> orig_to_new;
    int ctr = 0;

    // Find all rows that contain nonzeros
    for (std::vector<int>::iterator it = idx1.begin(); it != idx1.end(); ++it)
    {
        row_set.insert(*it);
    }

    // Condense matrix, removing zero rows, by creating map of original row
    // index to new (condensed) row index
    for (std::set<int>::iterator it = row_set.begin(); 
            it != row_set.end(); ++it)
    {
        orig_to_new[*it] = row_list.size();
        row_list.push_back(*it);
    }

    // Resize matrix to remove zero rows
    n_rows = row_set.size();

    // Map original row to new condensed row
    for (std::vector<int>::iterator it = idx1.begin(); it != idx1.end(); ++it)
    {
        *it = orig_to_new[*it];
    }
}

/**************************************************************
*****   COOMatrix Condense Columns
**************************************************************
***** Removes zero columns from the matrix, and initializes 
***** col_list, which points from new column index to original 
***** column index.
**************************************************************/
void COOMatrix::condense_cols()
{
    std::set<int> col_set;
    std::map<int, int> orig_to_new;
    int ctr = 0;

    // Find all cols that contain nonzeros
    for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
    {
        col_set.insert(*it);
    }

    // Condense matrix, removing zero cols, by creating map of original col
    // index to new (condensed) col index
    for (std::set<int>::iterator it = col_set.begin(); 
            it != col_set.end(); ++it)
    {
        orig_to_new[*it] = col_list.size();
        col_list.push_back(*it);
    }

    // Resize matrix to remove zero cols
    n_cols = col_set.size();

    // Map original col to new condensed col
    for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
    {
        *it = orig_to_new[*it];
    }
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
    // Create permutation vector p
    std::vector<int> p(nnz);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](int i, int j){ 
            if (idx1[i] == idx1[j])
                return idx2[i] < idx2[j];
            else
                return idx1[i] < idx1[j];
        });

    // Permute all vectors (rows, cols, data) 
    // according to p
    std::vector<bool> done(nnz);
    for (int i = 0; i < nnz; i++)
    {
        if (done[i]) continue;

        done[i] = true;
        int prev_j = i;
        int j = p[i];
        while (i != j)
        {
            std::swap(idx1[prev_j], idx1[j]);
            std::swap(idx2[prev_j], idx2[j]);
            std::swap(vals[prev_j], vals[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }

    // Remove duplicates (sum together)
    int prev_row = idx1[0];
    int prev_col = idx2[0];
    int ctr = 1;
    for (int i = 1; i < nnz; i++)
    {
        int row = idx1[i];
        int col = idx2[i];
        double val = vals[i];
        if (row == prev_row && col == prev_col)
        {
            vals[ctr-1] += val;
        }
        else if (ctr != i)
        {
            idx1[ctr] = row;
            idx2[ctr] = col;
            vals[ctr] = val;
            ctr++;
        }
    }
    nnz = ctr;

    // Move diagonal entry to first in row
    int row_start = 0;
    prev_row = 0;
    for (int i = 0; i < nnz; i++)
    {
        int row = idx1[i];
        int col = idx2[i];
        if (row != prev_row)
        {
            prev_row = row;
            row_start = i;
        }
        else if (row == col)
        {
            double tmp = vals[i];
            for (int j = i; j > row_start; j--)
            {
                idx2[j] = idx2[j-1];
                vals[j] = vals[j-1];
            }
            idx2[row_start] = row;
            vals[row_start] = tmp;
        }
    }
}

/**************************************************************
*****  COOMatrix Apply Function
**************************************************************
***** Apply function, passed as a paramter, to every nonzero
***** in the COO Matrix.
*****
***** Parameters
***** -------------
***** func_ptr : std::function<void(int, int, double)>
*****    Function that should be applied to every nonzero in 
*****    matrix.  This function needs to know only the row,
*****    column, and value of every nonzero.
**************************************************************/
void COOMatrix::apply_func(std::function<void(int, int, double)> func_ptr)
{
    for (int i = 0; i < nnz; i++)
    {
        int row = idx1[i];
        int col = idx2[i];
        double val = vals[i];
        func_ptr(row, col, val);
    }
} 

/**************************************************************
*****  COOMatrix Apply Function
**************************************************************
***** Apply function, passed as a paramter, to every nonzero
***** in the COO Matrix.  For an example, look at the method mult.
*****
***** Parameters
***** -------------
***** func_ptr : std::function<void(int, int, double, double*, double*)>
*****    Function that should be applied to every nonzero in 
*****    matrix.  This function needs to know the row,
*****    column, and value of every nonzero, as well as the two
*****    double* values passed as arguments.
***** xd : double* 
*****    Array of doubles that are used in function
***** bd : double*
*****    Array of doubles that are used in function
**************************************************************/
void COOMatrix::apply_func(Vector& xd, Vector& bd, 
        std::function<void(int, int, double, Vector&, Vector&)> func_ptr)
{
    for (int i = 0; i < nnz; i++)
    {
        int row = idx1[i];
        int col = idx2[i];
        double val = vals[i];
        func_ptr(row, col, val, xd, bd);
    }
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

void CSRMatrix::copy(const COOMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.resize(n_rows + 1, 0);
    if (nnz)
    {
        idx2.resize(nnz);
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
    std::vector<int> ctr;
    if (n_rows)
    {
    	ctr.resize(n_rows, 0);
    }
    for (int i = 0; i < nnz; i++)
    {
        int row = A->idx1[i];
        int col = A->idx2[i];
        double val = A->vals[i];
        int index = idx1[row] + ctr[row]++;
        idx2[index] = col;
        vals[index] = val;
    }
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}
void CSRMatrix::copy(const CSRMatrix* A)
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
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}
void CSRMatrix::copy(const CSCMatrix* A)
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
    std::vector<int> ctr(n_rows, 0);
    for (int i = 0; i < A->n_cols; i++)
    {
        int col_start = A->idx1[i];
        int col_end = A->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row = A->idx2[j];
            int idx = idx1[row] + ctr[row]++;
            idx2[idx] = i;
            vals[idx] = A->vals[j];
        }
    }
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}

/**************************************************************
*****   CSRMatrix Condense Rows
**************************************************************
***** Removes zero rows from the matrix, and initializes 
***** row_list, which points from new row index to original 
***** row index.
**************************************************************/
void CSRMatrix::condense_rows()
{
    std::vector<int> orig_to_new;
    orig_to_new.resize(n_rows);
    row_list.reserve(n_rows);
    int ctr = 0;

    for (int i = 0; i < n_rows; i++)
    {
        int row_size = idx1[i+1] - idx1[i];
        if (row_size)
        {
            orig_to_new[i] = row_list.size();
            row_list.push_back(i);
        }
    }

    for (std::vector<int>::iterator it = row_list.begin(); 
            it != row_list.end(); ++it)
    {
        idx1[ctr+1] = idx1[*it + 1];
        ctr++;
    }
    n_rows = row_list.size();
}

/**************************************************************
*****   CSRMatrix Condense Columns
**************************************************************
***** Removes zero columns from the matrix, and initializes 
***** col_list, which points from new column index to original 
***** column index.
**************************************************************/
void CSRMatrix::condense_cols()
{
    std::set<int> col_set;
    std::map<int, int> orig_to_new;
    int ctr = 0;

    // Find all cols that contain nonzeros
    for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
    {
        col_set.insert(*it);
    }

    // Condense matrix, removing zero cols, by creating map of original col
    // index to new (condensed) col index
    for (std::set<int>::iterator it = col_set.begin(); 
            it != col_set.end(); ++it)
    {
        orig_to_new[*it] = col_list.size();
        col_list.push_back(*it);
    }

    // Resize matrix to remove zero cols
    n_cols = col_set.size();

    // Map original col to new condensed col
    for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
    {
        *it = orig_to_new[*it];
    }
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
    // Sort the columns of each row (and data accordingly) and remove
    // duplicates (summing values together)
    int orig_row_start = idx1[0];
    for (int row = 0; row < n_rows; row++)
    {
        int new_row_start = idx1[row];
        int orig_row_end = idx1[row+1];
        int row_size = orig_row_end - orig_row_start;
        if (row_size == 0) 
        {
            orig_row_start = orig_row_end;
            idx1[row+1] = idx1[row];
            continue;
        }

        // Create permutation vector p for row
        std::vector<int> p(row_size);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(),
                [&](int i, int j)
                { 
                    return idx2[i+orig_row_start] < idx2[j+orig_row_start];
                });


        // Permute columns and data according to p
        std::vector<bool> done(row_size);
        for (int i = 0; i < row_size; i++)
        {
            if (done[i]) continue;

            done[i] = true;
            int prev_j = i;
            int j = p[i];
            while (i != j)
            {
                std::swap(idx2[prev_j + orig_row_start], idx2[j + orig_row_start]);
                std::swap(vals[prev_j + orig_row_start], vals[j + orig_row_start]);
                done[j] = true;
                prev_j = j;
                j = p[j];
            }
        }

        // Remove Duplicates
        int col = idx2[orig_row_start];
        double val = vals[orig_row_start];
        idx2[new_row_start] = col;
        vals[new_row_start] = val;
        int prev_col = col;
        int ctr = 1;
        for (int j = orig_row_start + 1; j < orig_row_end; j++)
        {
            col = idx2[j];
            val = vals[j];
            if (col == prev_col)
            {
                vals[ctr - 1 + new_row_start] += val;
            }
            else
            {
                idx2[ctr + new_row_start] = col;
                vals[ctr + new_row_start] = val;
                ctr++;
                prev_col = col;
            }
        }
        orig_row_start = orig_row_end;
        idx1[row+1] = idx1[row] + ctr;
    }

    // Move diagonal values to beginning of each row
    for (int i = 0; i < n_rows; i++)
    {
        int row_start = idx1[i];
        int row_end = idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            int col = idx2[j];
            if (col == i)
            {
                double tmp = vals[j];
                for (int k = j; k > row_start; k--)
                {
                    idx2[k] = idx2[k-1];
                    vals[k] = vals[k-1];
                }
                idx2[row_start] = i;
                vals[row_start] = tmp;
                break;
            }
        }
    }
}

/**************************************************************
*****  CSRMatrix Apply Function
**************************************************************
***** Apply function, passed as a paramter, to every nonzero
***** in the CSR Matrix.
*****
***** Parameters
***** -------------
***** func_ptr : std::function<void(int, int, double)>
*****    Function that should be applied to every nonzero in 
*****    matrix.  This function needs to know only the row,
*****    column, and value of every nonzero.
**************************************************************/
void CSRMatrix::apply_func(std::function<void(int, int, double)> func_ptr)
{
    for (int i = 0; i < n_rows; i++)
    {
        int row_start = idx1[i];
        int row_end = idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            func_ptr(i, idx2[j], vals[j]);
        }
    }
}

/**************************************************************
*****  CSRMatrix Apply Function
**************************************************************
***** Apply function, passed as a paramter, to every nonzero
***** in the CSR Matrix.  For an example, look at the method mult.
*****
***** Parameters
***** -------------
***** func_ptr : std::function<void(int, int, double, double*, double*)>
*****    Function that should be applied to every nonzero in 
*****    matrix.  This function needs to know the row,
*****    column, and value of every nonzero, as well as the two
*****    double* values passed as arguments.
***** xd : double* 
*****    Array of doubles that are used in function
***** bd : double*
*****    Array of doubles that are used in function
**************************************************************/
void CSRMatrix::apply_func(Vector& xd, Vector& bd, 
        std::function<void(int, int, double, Vector&, Vector&)> func_ptr)
{
    for (int i = 0; i < n_rows; i++)
    {
        int row_start = idx1[i];
        int row_end = idx1[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            func_ptr(i, idx2[j], vals[j], xd, bd);
        }
    }
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

void CSCMatrix::copy(const COOMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    idx1.resize(n_cols + 1);
    if (nnz)
    {
        idx2.resize(nnz);
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
    std::vector<int> ctr;
    if (n_cols)
    {
        ctr.resize(n_cols, 0);
    }
    for (int i = 0; i < A->nnz; i++)
    {
        int row = A->idx1[i];
        int col = A->idx2[i];
        double val = A->vals[i];
        int index = idx1[col] + ctr[col]++;
        idx2[index] = row;
        vals[index] = val;
    }        

    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }    
}
void CSCMatrix::copy(const CSRMatrix* A)
{
    n_rows = A->n_rows;
    n_cols = A->n_cols;
    nnz = A->nnz;

    // Resize vectors to appropriate dimensions
    idx1.resize(A->n_cols + 1);
    if (A->nnz)
    {
        idx2.resize(A->nnz);
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
        std::vector<int> ctr(A->n_cols, 0);
        for (int i = 0; i < A->n_rows; i++)
        {
            int row_start = A->idx1[i];
            int row_end = A->idx1[i+1];
            for (int j = row_start; j < row_end; j++)
            {
                int col = A->idx2[j];
                int idx = idx1[col] + ctr[col]++;
                idx2[idx] = i;
                vals[idx] = A->vals[j];
            }
        }
    }

    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}

void CSCMatrix::copy(const CSCMatrix* A)
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
    
    if (A->col_list.size())
    {
        col_list.resize(A->col_list.size());
        for (int i = 0; i < col_list.size(); i++)
        {
            col_list[i] = A->col_list[i];
        }
    }

    if (A->row_list.size())
    {
        row_list.resize(A->row_list.size());
        for (int i = 0; i < row_list.size(); i++)
        {
            row_list[i] = A->row_list[i];
        }
    }
}


/**************************************************************
*****   CSCMatrix Condense Rows
**************************************************************
***** Removes zero rows from the matrix, and initializes 
***** row_list, which points from new row index to original 
***** row index.
**************************************************************/
void CSCMatrix::condense_rows()
{
    std::set<int> row_set;
    std::map<int, int> orig_to_new;
    int ctr = 0;

    // Find all rows that contain nonzeros
    for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
    {
        row_set.insert(*it);
    }

    // Condense matrix, removing zero rows, by creating map of original row
    // index to new (condensed) row index
    for (std::set<int>::iterator it = row_set.begin(); 
            it != row_set.end(); ++it)
    {
        orig_to_new[*it] = row_list.size();
        row_list.push_back(*it);
    }

    // Resize matrix to remove zero rows
    n_rows = row_set.size();

    // Map original row to new condensed row
    for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
    {
        *it = orig_to_new[*it];
    }
}

/**************************************************************
*****   CSCMatrix Condense Columns
**************************************************************
***** Removes zero columns from the matrix, and initializes 
***** col_list, which points from new column index to original 
***** column index.
**************************************************************/
void CSCMatrix::condense_cols()
{
    std::vector<int> orig_to_new;
    orig_to_new.resize(n_cols);
    col_list.reserve(n_cols);
    int ctr = 0;

    for (int i = 0; i < n_cols; i++)
    {
        int col_size = idx1[i+1] - idx1[i];
        if (col_size)
        {
            orig_to_new[i] = col_list.size();
            col_list.push_back(i);
        }
    }

    for (std::vector<int>::iterator it = col_list.begin(); 
            it != col_list.end(); ++it)
    {
        idx1[ctr+1] = idx1[*it + 1];
        ctr++;
    }
    n_cols = col_list.size();
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
    // Sort the columns of each row (and data accordingly) and remove
    // duplicates (summing values together)
    int orig_col_start = idx1[0];
    for (int col = 0; col < n_cols; col++)
    {
        int new_col_start = idx1[col];
        int orig_col_end = idx1[col+1];
        int col_size = orig_col_end - orig_col_start;
        if (col_size == 0) 
        {
            orig_col_start = orig_col_end;
            idx1[col+1] = idx1[col];
            continue;
        }

        // Create permutation vector p for row
        std::vector<int> p(col_size);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(),
                [&](int i, int j)
                { 
                    return idx2[i + orig_col_start] < idx2[j + orig_col_start];
                });

        // Permute columns and data according to p
        std::vector<bool> done(col_size);
        for (int i = 0; i < col_size; i++)
        {
            if (done[i]) continue;

            done[i] = true;
            int prev_j = i;
            int j = p[i];
            while (i != j)
            {
                std::swap(idx2[prev_j + orig_col_start], idx2[j + orig_col_start]);
                std::swap(vals[prev_j + orig_col_start], vals[j + orig_col_start]);
                done[j] = true;
                prev_j = j;
                j = p[j];
            }
        }

        // Remove Duplicates
        int row = idx2[orig_col_start];
        double val = vals[orig_col_start];
        idx2[new_col_start] = row;
        vals[new_col_start] = val;
        int prev_row = row;
        int ctr = 1;
        for (int j = orig_col_start + 1; j < orig_col_end; j++)
        {
            row = idx2[j];
            val = vals[j];
            if (row == prev_row)
            {
                vals[ctr - 1 + new_col_start] += val;
            }
            else
            {
                idx2[ctr + new_col_start] = row;
                vals[ctr + new_col_start] = val;
                ctr++;
                prev_row = row;
            }
        }
        orig_col_start = orig_col_end;
        idx1[col+1] = idx1[col] + ctr;
    }
        
    // Move diagonal values to beginning of each column
    for (int i = 0; i < n_cols; i++)
    {
        int col_start = idx1[i];
        int col_end = idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            int row = idx2[j];
            if (row == i)
            {
                double tmp = vals[j];
                for (int k = j; k > col_start; k--)
                {
                    idx2[k] = idx2[k-1];
                    vals[k] = vals[k-1];
                }
                idx2[col_start] = i;
                vals[col_start] = tmp;
                break;
            }
        }
    }
}

/**************************************************************
*****  CSCMatrix Apply Function
**************************************************************
***** Apply function, passed as a paramter, to every nonzero
***** in the CSC Matrix.
*****
***** Parameters
***** -------------
***** func_ptr : std::function<void(int, int, double)>
*****    Function that should be applied to every nonzero in 
*****    matrix.  This function needs to know only the row,
*****    column, and value of every nonzero.
**************************************************************/
void CSCMatrix::apply_func(std::function<void(int, int, double)> func_ptr)
{
    for (int i = 0; i < n_cols; i++)
    {
        int col_start = idx1[i];
        int col_end = idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            func_ptr(idx2[j], i, vals[j]);
        }
    }
}

/**************************************************************
*****  CSCMatrix Apply Function
**************************************************************
***** Apply function, passed as a paramter, to every nonzero
***** in the CSC Matrix.  For an example, look at the method mult.
*****
***** Parameters
***** -------------
***** func_ptr : std::function<void(int, int, double, double*, double*)>
*****    Function that should be applied to every nonzero in 
*****    matrix.  This function needs to know the row,
*****    column, and value of every nonzero, as well as the two
*****    double* values passed as arguments.
***** xd : double* 
*****    Array of doubles that are used in function
***** bd : double*
*****    Array of doubles that are used in function
**************************************************************/
void CSCMatrix::apply_func(Vector& xd, Vector& bd, 
        std::function<void(int, int, double, Vector&, Vector&)> func_ptr)
{
    for (int i = 0; i < n_cols; i++)
    {
        int col_start = idx1[i];
        int col_end = idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            func_ptr(idx2[j], i, vals[j], xd, bd);
        }
    }
}  


