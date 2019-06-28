#include "core/matrix.hpp"
#include "omp.h"

using namespace raptor;
aligned_vector<double>& form_new(const CSRMatrix* A, const CSRMatrix* B, 
        CSRMatrix** C_ptr, aligned_vector<double>& A_vals)
{
    CSRMatrix* C = new CSRMatrix(A->n_rows, B->n_cols);
    *C_ptr = C;
    return C->vals;
}
aligned_vector<double*>& form_new(const CSRMatrix* A, const CSRMatrix* B, 
        CSRMatrix** C_ptr, aligned_vector<double*>& A_vals)
{
    BSRMatrix* C = new BSRMatrix(A->n_rows, B->n_cols, 
            A->b_rows, B->b_cols);
    *C_ptr = C;
    return C->block_vals;
}
aligned_vector<double>& form_new(const CSCMatrix* A, const CSRMatrix* B,
        CSRMatrix** C_ptr, aligned_vector<double>& A_vals)
{
    CSRMatrix* C = new CSRMatrix(A->n_cols, B->n_cols);
    *C_ptr = C;
    return C->vals;
}
aligned_vector<double*>& form_new(const CSCMatrix* A, const CSRMatrix* B,
        CSRMatrix** C_ptr, aligned_vector<double*>& A_vals)
{
    BSRMatrix* C = new BSRMatrix(A->n_cols, B->n_cols,
            A->b_cols, B->b_cols);
    *C_ptr = C;
    return C->block_vals;
}

void init_sums(aligned_vector<double>& sums, int size, int b_size)
{
    sums.resize(size, 0);
}
void init_sums(aligned_vector<double*>& sums, int size, int b_size)
{
    for (int i = 0; i < size; i++)
    {
        sums.emplace_back(new double[b_size]);
        for (int j = 0; j < b_size; j++)
            sums[i][j] = 0.0;
    }
}

void zero_sum(double* sum, int b_size)
{
    *sum = 0;
}
void zero_sum(double** sum, int b_size)
{
    (*sum) = new double[b_size];
    for (int i = 0; i < b_size; i++)
        (*sum)[i] = 0;
}

void finalize_sums(aligned_vector<double>& sums)
{
    return;
}
void finalize_sums(aligned_vector<double*>& sums)
{
    for (aligned_vector<double*>::iterator it = sums.begin();
            it != sums.end(); ++it)
        delete[] *it;
}

/*template <typename T>
CSRMatrix* spgemm_helper(const CSRMatrix* A, const CSRMatrix* B, 
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals,
        int* B_to_C = NULL)
{
    aligned_vector<int> next(B->n_cols, -1);
    aligned_vector<T> sums;
    init_sums(sums, B->n_cols, B->b_size);

    CSRMatrix* C = NULL;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*A->nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        int head = -2;
        int length = 0;
        int row_start_A = A->idx1[i];
        int row_end_A = A->idx1[i+1];
        for (int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = A->idx2[j];
            T val_A = A_vals[j];
            int row_start_B = B->idx1[col_A];
            int row_end_B = B->idx1[col_A+1];
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B->idx2[k];
                A->mult_vals(val_A, B_vals[k], &sums[col_B],
                        A->b_rows, B->b_cols, A->b_cols);
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
            double val = A->abs_val(sums[head]);
            if (val > zero_tol)
            {
                if (B_to_C) 
                {
                    C->idx2.emplace_back(B_to_C[head]);
                }
                else
                {
                    C->idx2.emplace_back(head);
                }
                C_vals.emplace_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            zero_sum(&sums[tmp], A->b_size);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    finalize_sums(sums);

    return C;
}*/

template <typename T>
CSRMatrix* spgemm_helper(const CSRMatrix* A, const CSRMatrix* B, 
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals,
        int* B_to_C = NULL)
{
    aligned_vector<int> next(B->n_cols, -1);
    aligned_vector<T> sums;
    init_sums(sums, B->n_cols, B->b_size);

    CSRMatrix* C = NULL;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*A->nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        int head = -2;
        int length = 0;
        int row_start_A = A->idx1[i];
        int row_end_A = A->idx1[i+1];

        for (int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = A->idx2[j];
            T val_A = A_vals[j];
            int row_start_B = B->idx1[col_A];
            int row_end_B = B->idx1[col_A+1];
          
            int col_B; 
            if ((row_end_B - row_start_B) % 3 == 0)
            { 
                for (int k = row_start_B; k < row_end_B; k+=3)
                {
                    col_B = B->idx2[k];
                    A->mult_vals(val_A, B_vals[k], &sums[col_B],
                            A->b_rows, B->b_cols, A->b_cols);
                    col_B = B->idx2[k+1];
                    A->mult_vals(val_A, B_vals[k+1], &sums[col_B],
                            A->b_rows, B->b_cols, A->b_cols);
                    col_B = B->idx2[k+2];
                    A->mult_vals(val_A, B_vals[k+2], &sums[col_B],
                            A->b_rows, B->b_cols, A->b_cols);
                }
            }
            else
            {
                for (int k = row_start_B; k < row_end_B; k++)
                {
                    int col_B = B->idx2[k];
                    A->mult_vals(val_A, B_vals[k], &sums[col_B],
                            A->b_rows, B->b_cols, A->b_cols);
                }
            }
            
            for (int k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B->idx2[k];
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
            double val = A->abs_val(sums[head]);
            if (val > zero_tol)
            {
                if (B_to_C) 
                {
                    C->idx2.emplace_back(B_to_C[head]);
                }
                else
                {
                    C->idx2.emplace_back(head);
                }
                C_vals.emplace_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            zero_sum(&sums[tmp], A->b_size);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    finalize_sums(sums);

    return C;
}

/*template <typename T>
CSRMatrix* spgemm_helper(const CSRMatrix* A, const CSRMatrix* B, 
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals,
        int* B_to_C = NULL)
{
    // THIS SPGEMM EDIT FOR 598
    aligned_vector<int> next(B->n_cols, -1);
    aligned_vector<T> sums;
    init_sums(sums, B->n_cols, B->b_size);

    CSRMatrix* C = NULL;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*A->nnz);

    C->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        int head = -2;
        int length = 0;
        int row_start_A = A->idx1[i];
        int row_end_A = A->idx1[i+1];
        for (int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = A->idx2[j];
            T val_A = A_vals[j];
            int row_start_B = B->idx1[col_A];
            int row_end_B = B->idx1[col_A+1];

            if (((row_end_B - row_start_B) % 9) == 0)
            {
            for (int k = row_start_B; k < row_end_B; k+=9)
            {
                int col_B1 = B->idx2[k];
                A->mult_vals(val_A, B_vals[k], &sums[col_B1],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B2 = B->idx2[k+1];
                A->mult_vals(val_A, B_vals[k+1], &sums[col_B2],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B3 = B->idx2[k+2];
                A->mult_vals(val_A, B_vals[k+2], &sums[col_B3],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B4 = B->idx2[k+3];
                A->mult_vals(val_A, B_vals[k+3], &sums[col_B4],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B5 = B->idx2[k+4];
                A->mult_vals(val_A, B_vals[k+4], &sums[col_B5],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B6 = B->idx2[k+5];
                A->mult_vals(val_A, B_vals[k+5], &sums[col_B6],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B7 = B->idx2[k+6];
                A->mult_vals(val_A, B_vals[k+6], &sums[col_B7],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B8 = B->idx2[k+7];
                A->mult_vals(val_A, B_vals[k+7], &sums[col_B8],
                        A->b_rows, B->b_cols, A->b_cols);
                int col_B9 = B->idx2[k+8];
                A->mult_vals(val_A, B_vals[k+8], &sums[col_B9],
                        A->b_rows, B->b_cols, A->b_cols);
                    if (next[col_B1] == -1)
                    {
                        next[col_B1] = head;
                        head = col_B1;
                        length++;
                    }
                    if (next[col_B2] == -1)
                    {
                        next[col_B2] = head;
                        head = col_B2;
                        length++;
                    }
                    if (next[col_B3] == -1)
                    {
                        next[col_B3] = head;
                        head = col_B3;
                        length++;
                    }
                    if (next[col_B4] == -1)
                    {
                        next[col_B4] = head;
                        head = col_B4;
                        length++;
                    }
                    if (next[col_B5] == -1)
                    {
                        next[col_B5] = head;
                        head = col_B5;
                        length++;
                    }
                    if (next[col_B6] == -1)
                    {
                        next[col_B6] = head;
                        head = col_B6;
                        length++;
                    }
                    if (next[col_B7] == -1)
                    {
                        next[col_B7] = head;
                        head = col_B7;
                        length++;
                    }
                    if (next[col_B8] == -1)
                    {
                        next[col_B8] = head;
                        head = col_B8;
                        length++;
                    }
                    if (next[col_B9] == -1)
                    {
                        next[col_B9] = head;
                        head = col_B9;
                        length++;
                    }
            }
            }
            else
            {
            for (k = row_start_B; k < row_end_B; k++)
            {
                int col_B = B->idx2[k];
                if (next[col_B] == -1)
                {
                    next[col_B] = head;
                    head = col_B;
                    length++;
                }
            }
            
            }
        }
        for (int j = 0; j < length; j++)
        {
            double val = A->abs_val(sums[head]);
            if (val > zero_tol)
            {
                if (B_to_C) 
                {
                    C->idx2.emplace_back(B_to_C[head]);
                }
                else
                {
                    C->idx2.emplace_back(head);
                }
                C_vals.emplace_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            zero_sum(&sums[tmp], A->b_size);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    finalize_sums(sums);

    return C;
}*/

/*template <typename T>
CSRMatrix* spgemm_T_helper(const CSCMatrix* A, const CSRMatrix* B,
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals,
        int* C_map = NULL)
{
    CSRMatrix* C;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*B->nnz);

    aligned_vector<int> next(B->n_cols, -1); 
    aligned_vector<T> sums;
    init_sums(sums, B->n_cols, A->b_size);

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
            T val_AT = A_vals[j];
            int row_start = B->idx1[col_AT];
            int row_end = B->idx1[col_AT+1];
            for (int k = row_start; k < row_end; k++)
            {
                int col = B->idx2[k];
                A->mult_T_vals(val_AT, B_vals[k], &sums[col],
                        A->b_cols, B->b_cols, A->b_rows);
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
            if (A->abs_val(sums[head]) > zero_tol)
            {
                if (C_map)
                {
                    C->idx2.emplace_back(C_map[head]);
                }
                else
                {
                    C->idx2.emplace_back(head);
                }
                C_vals.emplace_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            zero_sum(&sums[tmp], A->b_size);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    finalize_sums(sums);

    return C;
}*/

template <typename T>
CSRMatrix* spgemm_T_helper(const CSCMatrix* A, const CSRMatrix* B,
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals,
        int* C_map = NULL)
{
    CSRMatrix* C;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*B->nnz);

    aligned_vector<int> next(B->n_cols, -1); 
    aligned_vector<T> sums;
    init_sums(sums, B->n_cols, A->b_size);

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
            T val_AT = A_vals[j];
            int row_start = B->idx1[col_AT];
            int row_end = B->idx1[col_AT+1];

            if ((row_end - row_start) % 3 == 0)
            {
                for (int k = row_start; k < row_end; k+=3)
                {
                    int col = B->idx2[k];
                    A->mult_T_vals(val_AT, B_vals[k], &sums[col],
                            A->b_cols, B->b_cols, A->b_rows);
                    col = B->idx2[k+1];
                    A->mult_T_vals(val_AT, B_vals[k+1], &sums[col],
                            A->b_cols, B->b_cols, A->b_rows);
                    col = B->idx2[k+2];
                    A->mult_T_vals(val_AT, B_vals[k+2], &sums[col],
                            A->b_cols, B->b_cols, A->b_rows);
                }
            }
            else{
                for (int k = row_start; k < row_end; k++)
                {
                    int col = B->idx2[k];
                    A->mult_T_vals(val_AT, B_vals[k], &sums[col],
                            A->b_cols, B->b_cols, A->b_rows);
                }
            }
            
            for (int k = row_start; k < row_end; k++)
            {
                int col = B->idx2[k];
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
            if (A->abs_val(sums[head]) > zero_tol)
            {
                if (C_map)
                {
                    C->idx2.emplace_back(C_map[head]);
                }
                else
                {
                    C->idx2.emplace_back(head);
                }
                C_vals.emplace_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            zero_sum(&sums[tmp], A->b_size);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    finalize_sums(sums);

    return C;
}

/*template <typename T>
CSRMatrix* spgemm_T_helper(const CSCMatrix* A, const CSRMatrix* B,
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals,
        int* C_map = NULL)
{
    // THIS TRANSPOSE EDIT FOR 598
    CSRMatrix* C;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*B->nnz);

    aligned_vector<int> next(B->n_cols, -1); 
    aligned_vector<T> sums;
    init_sums(sums, B->n_cols, A->b_size);

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
            T val_AT = A_vals[j];
            int row_start = B->idx1[col_AT];
            int row_end = B->idx1[col_AT+1];
            
            if (((row_end - row_start) % 9) == 0)
            {
                for (int k = row_start; k < row_end; k+=9)
                {
                    int col1 = B->idx2[k];
                    A->mult_T_vals(val_AT, B_vals[k], &sums[col1],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col2 = B->idx2[k+1];
                    A->mult_T_vals(val_AT, B_vals[k+1], &sums[col2],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col3 = B->idx2[k+2];
                    A->mult_T_vals(val_AT, B_vals[k+2], &sums[col3],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col4 = B->idx2[k+3];
                    A->mult_T_vals(val_AT, B_vals[k+3], &sums[col4],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col5 = B->idx2[k+4];
                    A->mult_T_vals(val_AT, B_vals[k+4], &sums[col5],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col6 = B->idx2[k+5];
                    A->mult_T_vals(val_AT, B_vals[k+5], &sums[col6],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col7 = B->idx2[k+6];
                    A->mult_T_vals(val_AT, B_vals[k+6], &sums[col7],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col8 = B->idx2[k+7];
                    A->mult_T_vals(val_AT, B_vals[k+7], &sums[col8],
                            A->b_cols, B->b_cols, A->b_rows);
                    int col9 = B->idx2[k+8];
                    A->mult_T_vals(val_AT, B_vals[k+8], &sums[col9],
                            A->b_cols, B->b_cols, A->b_rows);
                   
                    if (next[col1] == -1)
                    {
                        next[col1] = head;
                        head = col1;
                        length++;
                    }
                    if (next[col2] == -1)
                    {
                        next[col2] = head;
                        head = col2;
                        length++;
                    }
                    if (next[col3] == -1)
                    {
                        next[col3] = head;
                        head = col3;
                        length++;
                    }
                    if (next[col4] == -1)
                    {
                        next[col4] = head;
                        head = col4;
                        length++;
                    }
                    if (next[col5] == -1)
                    {
                        next[col5] = head;
                        head = col5;
                        length++;
                    }
                    if (next[col6] == -1)
                    {
                        next[col6] = head;
                        head = col6;
                        length++;
                    }
                    if (next[col7] == -1)
                    {
                        next[col7] = head;
                        head = col7;
                        length++;
                    }
                    if (next[col8] == -1)
                    {
                        next[col8] = head;
                        head = col8;
                        length++;
                    }
                    if (next[col9] == -1)
                    {
                        next[col9] = head;
                        head = col9;
                        length++;
                    }
                }
            }
            else
            {
                for (int k = row_start; k < row_end; k++)
                {
                    int col = B->idx2[k];
                    A->mult_T_vals(val_AT, B_vals[k], &sums[col],
                            A->b_cols, B->b_cols, A->b_rows);
                    if (next[col] == -1)
                    {
                        next[col] = head;
                        head = col;
                        length++;
                    }
                }
            }
        }

        for (int j = 0; j < length; j++)
        {
            if (A->abs_val(sums[head]) > zero_tol)
            {
                if (C_map)
                {
                    C->idx2.emplace_back(C_map[head]);
                }
                else
                {
                    C->idx2.emplace_back(head);
                }
                C_vals.emplace_back(sums[head]);
            }
            int tmp = head;
            head = next[head];
            next[tmp] = -1;
            zero_sum(&sums[tmp], A->b_size);
        }
        C->idx1[i+1] = C->idx2.size();
    }
    C->nnz = C->idx2.size();

    finalize_sums(sums);

    return C;
}*/


CSRMatrix* Matrix::mult(CSRMatrix* B, int* B_to_C)
{
    return spgemm(B, B_to_C);
}
CSRMatrix* Matrix::mult(CSCMatrix* B, int* B_to_C)
{
    CSRMatrix* B_csr = B->to_CSR();
    CSRMatrix* C = spgemm(B_csr, B_to_C);
    delete B_csr;
    return C;
}
CSRMatrix* Matrix::mult(COOMatrix* B, int* B_to_C)
{
    CSRMatrix* B_csr = B->to_CSR();
    CSRMatrix* C = spgemm(B_csr, B_to_C);
    delete B_csr;
    return C;
}

CSRMatrix* Matrix::mult_T(CSCMatrix* A, int* C_map)
{
    return spgemm_T(A, C_map);
}
CSRMatrix* Matrix::mult_T(CSRMatrix* A, int* C_map)
{
    CSCMatrix* A_csc = A->to_CSC();
    CSRMatrix* C = spgemm_T(A_csc, C_map);
    delete A_csc;
    return C;
}
CSRMatrix* Matrix::mult_T(COOMatrix* A, int* C_map)
{
    CSCMatrix* A_csc = A->to_CSC();
    CSRMatrix* C = spgemm_T(A_csc, C_map);
    delete A_csc;
    return C;
}

CSRMatrix* CSRMatrix::spgemm(CSRMatrix* B, int* B_to_C)
{
    return spgemm_helper(this, B, vals, B->vals, B_to_C);
}
BSRMatrix* BSRMatrix::spgemm(CSRMatrix* B, int* B_to_C)
{
    BSRMatrix* B_bsr = (BSRMatrix*) B;
    return (BSRMatrix*) spgemm_helper(this, B_bsr, block_vals, 
            B_bsr->block_vals, B_to_C);
}
CSRMatrix* COOMatrix::spgemm(CSRMatrix* B, int* B_to_C)
{
    CSRMatrix* A_csr = to_CSR();
    CSRMatrix* C = spgemm_helper(A_csr, B, A_csr->vals, B->vals, 
            B_to_C);
    delete A_csr;
    return C;
}
BSRMatrix* BCOOMatrix::spgemm(CSRMatrix* B, int* B_to_C)
{
    BSRMatrix* A_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* B_bsr = (BSRMatrix*) B;
    BSRMatrix* C = (BSRMatrix*) spgemm_helper(A_bsr, B_bsr, 
            A_bsr->block_vals, B_bsr->block_vals, B_to_C);
    delete A_bsr;
    return C;
}
CSRMatrix* CSCMatrix::spgemm(CSRMatrix* B, int* B_to_C)
{
    CSRMatrix* A_csr = to_CSR();
    CSRMatrix* C = spgemm_helper(A_csr, B, A_csr->vals, B->vals,
            B_to_C);
    delete A_csr;
    return C;
}
BSRMatrix* BSCMatrix::spgemm(CSRMatrix* B, int* B_to_C)
{
    BSRMatrix* A_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* B_bsr = (BSRMatrix*) B;
    BSRMatrix* C = (BSRMatrix*) spgemm_helper(A_bsr, B_bsr, 
            A_bsr->block_vals, B_bsr->block_vals, B_to_C);
    delete A_bsr;
    return C;
}


CSRMatrix* CSRMatrix::spgemm_T(CSCMatrix* A, int* C_map)
{
    return spgemm_T_helper(A, this, A->vals, vals, C_map);
}
BSRMatrix* BSRMatrix::spgemm_T(CSCMatrix* A, int* C_map)
{
    BSCMatrix* A_bsc = (BSCMatrix*) A;
    return (BSRMatrix*) spgemm_T_helper(A_bsc, this, 
            A_bsc->block_vals, block_vals, C_map);
}
CSRMatrix* COOMatrix::spgemm_T(CSCMatrix* A, int* C_map)
{
    CSRMatrix* B_csr = to_CSR();
    CSRMatrix* C = spgemm_T_helper(A, B_csr, A->vals, 
            B_csr->vals, C_map);
    delete B_csr;
    return C;
}
BSRMatrix* BCOOMatrix::spgemm_T(CSCMatrix* A, int* C_map)
{
    BSCMatrix* A_bsc = (BSCMatrix*) A;
    BSRMatrix* B_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* C = (BSRMatrix*) spgemm_T_helper(A_bsc, B_bsr, 
            A_bsc->block_vals, B_bsr->block_vals, C_map);
    delete B_bsr;
    return C;
}
CSRMatrix* CSCMatrix::spgemm_T(CSCMatrix* A, int* C_map)
{
    CSRMatrix* B_csr = to_CSR();
    CSRMatrix* C = spgemm_T_helper(A, B_csr, A->vals, 
            B_csr->vals, C_map);
    delete B_csr;
    return C;
}
BSRMatrix* BSCMatrix::spgemm_T(CSCMatrix* A, int* C_map)
{
    BSCMatrix* A_bsc = (BSCMatrix*) A;
    BSRMatrix* B_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* C = (BSRMatrix*) spgemm_T_helper(A_bsc, B_bsr, 
            A_bsc->block_vals, B_bsr->block_vals, C_map);
    delete B_bsr;
    return C;
}
