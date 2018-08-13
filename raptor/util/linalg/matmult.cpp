#include "core/matrix.hpp"

using namespace raptor;

CSRMatrix* CSRMatrix::spgemm(const CSRMatrix* B)
{
    aligned_vector<int> next(n_cols, -1);
    aligned_vector<double> sums(n_cols, 0);

    CSRMatrix* C = new CSRMatrix(n_rows, B->n_cols);
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    C->idx1[0] = 0;
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
            int row_start_B = B->idx1[col_A];
            int row_end_B = B->idx1[col_A+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B->idx2[k];
                sums[col_B]+= val_A*B->vals[k];
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
            if (fabs(sums[head]) > zero_tol)
            {
                C->idx2.push_back(head);
                C->vals.push_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    return C;
}

CSRMatrix* CSRMatrix::spgemm_T(const CSCMatrix* A)
{
    CSRMatrix* C = new CSRMatrix(A->n_cols, n_cols);
    C->idx2.reserve(1.5*nnz);
    C->vals.reserve(1.5*nnz);

    aligned_vector<int> next(A->n_rows, -1); 
    aligned_vector<double> sums(A->n_rows, 0);

    C->idx1[0] = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        int head = -2;
        int length = 0;
        int row_start_AT = A->idx1[i];
        int row_end_AT = A->idx1[i+1];
        for (int j = row_start_AT; j < row_end_AT; j++)
        {
            int col_AT = A->idx2[j];
            double val_AT = A->vals[j];
            int row_start = idx1[col_AT];
            int row_end = idx1[col_AT+1];
            for (int k = row_start; k < row_end; k++)
            {
                int col = idx2[k];
                sums[col]+= val_AT*vals[k];
                if (next[col] == -1)
                {
                    next[col] = head;
                    head = col;
                    length++;
                }
            }
        }
        for (int j = 0; j < length; j++)
        {
            if (fabs(sums[head]) > zero_tol)
            {
                C->idx2.push_back(head);
                C->vals.push_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            sums[tmp] = 0;
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    return C;
}

