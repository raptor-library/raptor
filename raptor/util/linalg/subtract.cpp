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
CSRMatrix* CSRMatrix::subtract(CSRMatrix* B)
{
    int idx, idx_B;
    int row_end, row_end_B;
    int col, col_B;

    int* col_ptr;
    int* col_B_ptr;

    std::vector<int> cols;
    std::vector<int> cols_B;

    CSRMatrix* C = new CSRMatrix(n_rows, n_cols);

    sort();
    B->sort();

    if (col_list.size() || B->col_list.size())
    {
        // Create C col_list to be union of col_list and B->col_list
        std::set<int> global_col_set;
        for (std::vector<int>::iterator it = col_list.begin(); 
                it != col_list.end(); ++it)
        {
            global_col_set.insert(*it);
        }
        for (std::vector<int>::iterator it = B->col_list.begin();
                it != B->col_list.end(); ++it)
        {
            global_col_set.insert(*it);
        }
        std::map<int, int> global_to_local_C;
        for (std::set<int>::iterator it = global_col_set.begin();
                it != global_col_set.end(); ++it)
        {
            global_to_local_C[*it] = C->col_list.size();
            C->col_list.push_back(*it);
        }

        // Map col_list values to C_col_list values
        if (col_list.size())
        {
            std::vector<int> map_to_C;
            map_to_C.reserve(col_list.size());
            if (idx2.size())
            {
                cols.reserve(idx2.size());
            }
            for (std::vector<int>::iterator it = col_list.begin(); 
                    it != col_list.end(); ++it)
            {
                map_to_C.push_back(global_to_local_C[*it]);
            }

            for (std::vector<int>::iterator it = idx2.begin(); it != idx2.end(); ++it)
            {
                cols.push_back(map_to_C[*it]);
            }
        }

        // Map B_col_list values to C_col_list values
        if (B->col_list.size())
        {
            std::vector<int> B_to_C;
            B_to_C.reserve(B->col_list.size());
            for (std::vector<int>::iterator it = B->col_list.begin();
                    it != B->col_list.end(); ++it)
            {
                B_to_C.push_back(global_to_local_C[*it]);
            }
            if (B->idx2.size())
            {
                cols_B.reserve(B->idx2.size());
            }
            for (std::vector<int>::iterator it = B->idx2.begin(); 
                    it != B->idx2.end(); ++it)
            {
                cols_B.push_back(B_to_C[*it]);
            }
        }

        C->resize(n_rows, global_col_set.size());

    }
    if (col_list.size())
    {
        col_ptr = cols.data();
    }
    else
    {
        col_ptr = idx2.data();
    }

    if (B->col_list.size())
    {
        col_B_ptr = cols_B.data();
    }
    else
    {
        col_B_ptr = B->idx2.data();
    }

    C->idx2.reserve(1.1*nnz);
    C->vals.reserve(1.1*nnz);

    // Note -- diagonal values are first, and then others
    C->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        idx = idx1[i];
        row_end = idx1[i+1];
        idx_B = B->idx1[i];
        row_end_B = B->idx1[i+1];

        double val = 0.0;
        // Get inital column
        if (idx < row_end)
        {
            col = col_ptr[idx];
            if (col == i)
            {
                val = vals[idx];
                idx++;
                if (idx < row_end)
                    col = col_ptr[idx];
                else
                    col = C->n_cols;
            }
        }
        else
            col = C->n_cols;

        // Get initial column of B
        if (idx_B < row_end_B)
        {
            col_B = col_B_ptr[idx_B];
            if (col_B == i)
            {
                val -= B->vals[idx_B];
                idx_B++;
                if (idx_B < row_end_B)
                    col_B = col_B_ptr[idx_B];
                else
                    col_B = C->n_cols;
            }
        }
        else
            col_B = C->n_cols;

        // If either col or col_b equals row, add val - val_b to C
        if (fabs(val) > zero_tol)
        {
            C->idx2.push_back(i);
            C->vals.push_back(val);
        }

        while (idx < row_end || idx_B < row_end_B)
        {
            // If columns are equal, add val-B.val
            if (col == col_B)
            {
                val = vals[idx] - B->vals[idx_B];
                if (fabs(val) > zero_tol)
                {
                    C->idx2.push_back(col);
                    C->vals.push_back(vals[idx] - B->vals[idx_B]);
                }
                // Increase index and find column of new index
                idx++;
                if (idx < row_end)
                    col = col_ptr[idx];
                else
                    col = C->n_cols;

                // Increase index of B and find column
                idx_B++;
                if (idx_B < row_end_B)
                    col_B = col_B_ptr[idx_B];
                else
                    col_B = C->n_cols;
            }

            // If column comes first, add val
            else if (col < col_B)
            {
                C->idx2.push_back(col);
                C->vals.push_back(vals[idx]);

                // Increase index and find column
                idx++;
                if (idx < row_end)
                    col = col_ptr[idx];
                else
                    col = C->n_cols;

            }

            // If B.column comes first, add -B.val
            else
            {
                C->idx2.push_back(col_B);
                C->vals.push_back(-(B->vals[idx_B]));
                
                // Increase index of B and find column
                idx_B++;
                if (idx_B < row_end_B)
                    col_B = col_B_ptr[idx_B];
                else
                    col_B = C->n_cols;
            }

        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    return C;
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
Matrix* Matrix::subtract(Matrix* B)
{
    if (format() == CSR && B->format() == CSR)
    {
        CSRMatrix* C = ((CSRMatrix*)this)->subtract((CSRMatrix*) B);
        return C;
    }

    printf("Subtraction not implemented for these matrix types...\n");
    return NULL;
}


