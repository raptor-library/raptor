#include "core/seq/matrix.hpp"

using namespace raptor;

/**************************************************************
*****   Matrix-Matrix Multiply (C = A*B)
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
void Matrix::mult(CSRMatrix& B, CSRMatrix& C)
{
    if (format() == COO || format() == CSC)
    {
        printf("Not currently implemented for this type of matrix.\n");
        return;
    }

    std::vector<int> next(n_cols, -1);
    std::vector<double> sums(n_cols, 0);
    nnz = 0;
    C.idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        int head = -2;
        int length = 0;
        int row_start_A = idx1[i];
        int row_end_A = idx1[i+1];
        for (int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = idx2[j];
            double val_A = vals[j];
            int row_start_B = B.idx1[col_A];
            int row_end_B = B.idx1[col_A+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B.idx2[k];
                sums[col_B]+= val_A*B.vals[k];
                if (next[col_B] == -1)
                {
                    next[col_B] = head;
                    head = col_B;
                    length++;
                }
            }
        }
        for (int j = 0; j < length; j++)
        {
            if (sums[head] != 0)
            {
                C.idx2[nnz] = head;
                C.vals[nnz] = sums[head];
                nnz++;
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C.idx1[i+1] = nnz;
    }
}


