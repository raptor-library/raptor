#include "core/matrix.hpp"

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
    return C->vals;
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
    return C->vals;
}

void init_sums(aligned_vector<double>& sums, int size, int b_size)
{
    sums.resize(size, 0);
}
void init_sums(aligned_vector<double*>& sums, int size, int b_size)
{
    for (int i = 0; i < size; i++)
    {
        sums.push_back(new double[b_size]);
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

template <typename T>
CSRMatrix* spgemm_helper(const CSRMatrix* A, const CSRMatrix* B, 
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals)
{
    aligned_vector<int> next(A->n_cols, -1);
    aligned_vector<T> sums;
    init_sums(sums, A->n_cols, A->b_size);

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
            if (A->abs_val(sums[head]) > zero_tol)
            {
                C->idx2.push_back(head);
                C_vals.push_back(sums[head]);
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

template <typename T>
CSRMatrix* spgemm_T_helper(const CSCMatrix* A, const CSRMatrix* B,
        aligned_vector<T>& A_vals, aligned_vector<T>& B_vals)
{
    CSRMatrix* C;
    aligned_vector<T>& C_vals = form_new(A, B, &C, A_vals);
    C->reserve_size(1.5*B->nnz);

    aligned_vector<int> next(A->n_rows, -1); 
    aligned_vector<T> sums;
    init_sums(sums, A->n_rows, A->b_size);

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
                C->idx2.push_back(head);
                C_vals.push_back(sums[head]);
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


CSRMatrix* Matrix::mult(CSRMatrix* B)
{
    return spgemm(B);
}
CSRMatrix* Matrix::mult(CSCMatrix* B)
{
    CSRMatrix* B_csr = B->to_CSR();
    CSRMatrix* C = spgemm(B_csr);
    delete B_csr;
    return C;
}
CSRMatrix* Matrix::mult(COOMatrix* B)
{
    CSRMatrix* B_csr = B->to_CSR();
    CSRMatrix* C = spgemm(B_csr);
    delete B_csr;
    return C;
}

CSRMatrix* Matrix::mult_T(CSCMatrix* A)
{
    return spgemm_T(A);
}
CSRMatrix* Matrix::mult_T(CSRMatrix* A)
{
    CSCMatrix* A_csc = A->to_CSC();
    CSRMatrix* C = spgemm_T(A_csc);
    delete A_csc;
    return C;
}
CSRMatrix* Matrix::mult_T(COOMatrix* A)
{
    CSCMatrix* A_csc = A->to_CSC();
    CSRMatrix* C = spgemm_T(A_csc);
    delete A_csc;
    return C;
}

CSRMatrix* CSRMatrix::spgemm(CSRMatrix* B)
{
    return spgemm_helper(this, B, vals, B->vals);
}
BSRMatrix* BSRMatrix::spgemm(CSRMatrix* B)
{
    BSRMatrix* B_bsr = (BSRMatrix*) B;
    return (BSRMatrix*) spgemm_helper(this, B_bsr, vals, B_bsr->vals);
}
CSRMatrix* COOMatrix::spgemm(CSRMatrix* B)
{
    CSRMatrix* A_csr = to_CSR();
    CSRMatrix* C = spgemm_helper(A_csr, B, A_csr->vals, B->vals);
    delete A_csr;
    return C;
}
BSRMatrix* BCOOMatrix::spgemm(CSRMatrix* B)
{
    BSRMatrix* A_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* B_bsr = (BSRMatrix*) B;
    BSRMatrix* C = (BSRMatrix*) spgemm_helper(A_bsr, B_bsr, A_bsr->vals, B_bsr->vals);
    delete A_bsr;
    return C;
}
CSRMatrix* CSCMatrix::spgemm(CSRMatrix* B)
{
    CSRMatrix* A_csr = to_CSR();
    CSRMatrix* C = spgemm_helper(A_csr, B, A_csr->vals, B->vals);
    delete A_csr;
    return C;
}
BSRMatrix* BSCMatrix::spgemm(CSRMatrix* B)
{
    BSRMatrix* A_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* B_bsr = (BSRMatrix*) B;
    BSRMatrix* C = (BSRMatrix*) spgemm_helper(A_bsr, B_bsr, A_bsr->vals, B_bsr->vals);
    delete A_bsr;
    return C;
}


CSRMatrix* CSRMatrix::spgemm_T(CSCMatrix* A)
{
    return spgemm_T_helper(A, this, A->vals, vals);
}
BSRMatrix* BSRMatrix::spgemm_T(CSCMatrix* A)
{
    BSCMatrix* A_bsc = (BSCMatrix*) A;
    return (BSRMatrix*) spgemm_T_helper(A_bsc, this, A_bsc->vals, vals);
}
CSRMatrix* COOMatrix::spgemm_T(CSCMatrix* A)
{
    CSRMatrix* B_csr = to_CSR();
    CSRMatrix* C = spgemm_T_helper(A, B_csr, A->vals, B_csr->vals);
    delete B_csr;
    return C;
}
BSRMatrix* BCOOMatrix::spgemm_T(CSCMatrix* A)
{
    BSCMatrix* A_bsc = (BSCMatrix*) A;
    BSRMatrix* B_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* C = (BSRMatrix*) spgemm_T_helper(A_bsc, B_bsr, A_bsc->vals, B_bsr->vals);
    delete B_bsr;
    return C;
}
CSRMatrix* CSCMatrix::spgemm_T(CSCMatrix* A)
{
    CSRMatrix* B_csr = to_CSR();
    CSRMatrix* C = spgemm_T_helper(A, B_csr, A->vals, B_csr->vals);
    delete B_csr;
    return C;
}
BSRMatrix* BSCMatrix::spgemm_T(CSCMatrix* A)
{
    BSCMatrix* A_bsc = (BSCMatrix*) A;
    BSRMatrix* B_bsr = (BSRMatrix*) to_CSR();
    BSRMatrix* C = (BSRMatrix*) spgemm_T_helper(A_bsc, B_bsr, A_bsc->vals, B_bsr->vals);
    delete B_bsr;
    return C;
}
