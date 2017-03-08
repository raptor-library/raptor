#include "core/matrix.hpp"

using namespace raptor;

/**************************************************************
*****   Matrix-Matrix Subtraction (C = A-B)
**************************************************************
***** Multiplies the matrix times a matrix B, and writes the
***** result in matrix C.
*****
***** Parameters
***** -------------
***** B : CSRMatrix*
*****    Matrix by which to multiply the matrix 
***** C : Matrix*
*****    CSRMatrix in which to place solution
**************************************************************/
void Matrix::subtract(CSRMatrix& B, CSRMatrix& C)
{
    if (format() == COO || format() == CSC)
    {
        printf("Not currently implemented for this type of matrix.\n");
        return;
    }

    C.n_rows = n_rows;
    C.n_cols = n_cols;

    C.idx1.resize(n_rows + 1, 0);
    C.idx2.clear();
    C.idx2.reserve(2*nnz);
    C.vals.clear();
    C.vals.reserve(2*nnz);

    int ctr = 0;
    int idx, idx_b;
    int row_end, row_end_b;
    int col, col_b;

    // Note -- diagonal values are first, and then others
    C.idx1[0] = ctr;
    for (int i = 0; i < n_rows; i++)
    {
        idx = idx1[i];
        row_end = idx1[i+1];
        idx_b = B.idx1[i];
        row_end_b = B.idx1[i+1];

        // Get inital column
        if (idx < row_end)
        {
            col = idx2[idx];
        }
        else
            col = n_cols;

        // Get initial column of B
        if (idx_b < row_end_b)
        {
            col_b = B.idx2[idx_b];
        }
        else
            col_b = n_cols;

        // If either col or col_b equals row, add val - val_b to C
        double val = 0.0;
        if (col == i)
        {
            val = vals[idx];
            idx++;
            if (idx < row_end)
                col = idx2[idx];
            else
                col = n_cols;
        }
        if (col_b == i)
        {
            val -= B.vals[idx_b];
            idx_b++;
            if (idx_b < row_end_b)
                col_b = B.idx2[idx_b];
            else
                col_b = n_cols;
        }
        if (fabs(val) > zero_tol)
        {
            C.idx2.push_back(i);
            C.vals.push_back(val);
            ctr++;
        }

        while (idx < row_end || idx_b < row_end_b)
        {
            // If columns are equal, add val-B.val
            if (col == col_b)
            {
                val = vals[idx] - B.vals[idx_b];
                if (fabs(val) > zero_tol)
                {
                    C.idx2.push_back(col);
                    C.vals.push_back(vals[idx] - B.vals[idx_b]);
                    ctr++;
                }
                // Increase index and find column of new index
                idx++;
                if (idx < row_end)
                    col = idx2[idx];
                else
                    col = n_cols;

                // Increase index of B and find column
                idx_b++;
                if (idx_b < row_end_b)
                    col_b = B.idx2[idx_b];
                else
                    col_b = n_cols;

            }

            // If column comes first, add val
            else if (col < col_b)
            {
                C.idx2.push_back(col);
                C.vals.push_back(vals[idx]);
                ctr++;

                // Increase index and find column
                idx++;
                if (idx < row_end)
                    col = idx2[idx];
                else
                    col = n_cols;

            }

            // If B.column comes first, add -B.val
            else
            {
                C.idx2.push_back(col_b);
                C.vals.push_back(-(B.vals[idx_b]));
                ctr++;
                
                // Increase index of B and find column
                idx_b++;
                if (idx_b < row_end_b)
                    col_b = B.idx2[idx_b];
                else
                    col_b = n_cols;
            }

        }
        C.idx1[i+1] = ctr;
    }

    C.nnz = C.idx2.size();
}


/**************************************************************
*****   Matrix-Matrix Subtraction (C = A-B)
**************************************************************
***** Multiplies the matrix times a matrix B, and writes the
***** result in matrix C.
*****
***** Parameters
***** -------------
***** B : CSRMatrix*
*****    Matrix by which to multiply the matrix 
***** C : Matrix*
*****    CSRMatrix in which to place solution
**************************************************************/
void Matrix::subtract(CSCMatrix& B, CSCMatrix& C)
{
    if (format() == COO || format() == CSR)
    {
        printf("Not currently implemented for this type of matrix.\n");
        return;
    }

    C.n_rows = n_rows;
    C.n_cols = n_cols;

    C.idx1.resize(n_cols + 1, 0);
    C.idx2.clear();
    C.idx2.reserve(2*nnz);
    C.vals.clear();
    C.vals.reserve(2*nnz);

    int ctr = 0;
    int idx, idx_b;
    int col_end, col_end_b;
    int row, row_b;

    // Note -- diagonal values are first, and then others
    C.idx1[0] = ctr;
    for (int i = 0; i < n_cols; i++)
    {
        idx = idx1[i];
        col_end = idx1[i+1];
        idx_b = B.idx1[i];
        col_end_b = B.idx1[i+1];

        // Get inital column
        if (idx < col_end)
        {
            row = idx2[idx];
        }
        else
            row = n_rows;

        // Get initial column of B
        if (idx_b < col_end_b)
        {
            row_b = B.idx2[idx_b];
        }
        else
            row_b = n_rows;

        // If either col or col_b equals row, add val - val_b to C
        double val = 0.0;
        if (row == i)
        {
            val = vals[idx];
            idx++;
            if (idx < col_end)
                row = idx2[idx];
            else
                row = n_rows;
        }
        if (row_b == i)
        {
            val -= B.vals[idx_b];
            idx_b++;
            if (idx_b < col_end_b)
                row_b = B.idx2[idx_b];
            else
                row_b = n_rows;
        }
        if (fabs(val) > zero_tol)
        {
            C.idx2.push_back(i);
            C.vals.push_back(val);
            ctr++;
        }

        while (idx < col_end || idx_b < col_end_b)
        {
            // If columns are equal, add val-B.val
            if (row == row_b)
            {
                val = vals[idx] - B.vals[idx_b];
                if (fabs(val) > zero_tol)
                {
                    C.idx2.push_back(row);
                    C.vals.push_back(vals[idx] - B.vals[idx_b]);
                    ctr++;
                }
                // Increase index and find column of new index
                idx++;
                if (idx < col_end)
                    row = idx2[idx];
                else
                    row = n_rows;

                // Increase index of B and find column
                idx_b++;
                if (idx_b < col_end_b)
                    row_b = B.idx2[idx_b];
                else
                    row_b = n_rows;

            }

            // If column comes first, add val
            else if (row < row_b)
            {
                C.idx2.push_back(row);
                C.vals.push_back(vals[idx]);
                ctr++;

                // Increase index and find column
                idx++;
                if (idx < col_end)
                    row = idx2[idx];
                else
                    row = n_rows;

            }

            // If B.column comes first, add -B.val
            else
            {
                C.idx2.push_back(row_b);
                C.vals.push_back(-(B.vals[idx_b]));
                ctr++;
                
                // Increase index of B and find column
                idx_b++;
                if (idx_b < col_end_b)
                    row_b = B.idx2[idx_b];
                else
                    row_b = n_rows;
            }

        }
        C.idx1[i+1] = ctr;
    }

    C.nnz = C.idx2.size();
}


