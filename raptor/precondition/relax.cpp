// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "relax.hpp"

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

    void dgemv_(char* TRANS, const int* M, const int* N,
               double* alpha, double* A, const int* LDA, double* X,
               const int* INCX, double* beta, double* C, const int* INCY); 
}


namespace raptor {

/**
 * Block Matrix (BSR) Relaxation Methods
 **/


// Inverts block at address A
// Returns inverted block at address A_inv
void invert_block(double* A, int n)
{
    int info;
    int block_size = n*n;
    int *lu = new int[n];
    double* tmp = new double[block_size];

    dgetrf_(&n, &n, A, &n, lu, &info);
    dgetri_(&n, A, &n, lu, tmp, &block_size, &info);

    delete[] tmp;
    delete[] lu;
}

void block_relax_init(BSRMatrix* A, double** D_inv_ptr)
{
    double* D_inv = new double[A->n_rows*A->b_size]();
    double** bdata = (double**)(A->get_data());
    int row_start, row_end;

    // Invert all diagonal blocks of A (should really do this during AMG setup and store)
    for (int i = 0; i < A->n_rows; i++)
    {
        row_start = A->idx1[i];
        row_end = A->idx1[i+1];
        if (row_start == row_end) continue;
        for (int j = row_start; j < row_end; j++)
        {
            int col = A->idx2[j];
            if (i == col)
            {
                memcpy(&(D_inv[i*A->b_size]), bdata[j], A->b_size*sizeof(double));
                invert_block(&(D_inv[i*A->b_size]), A->b_rows);

                break;
            }
        }
    }

    *D_inv_ptr = D_inv;
}


void block_relax_free(double* A_inv)
{
    delete[] A_inv;
}

void jacobi_copy(Vector& tmp, Vector& x)
{
    memcpy(tmp.data(), x.data(), tmp.size()*sizeof(double));
}
void sor_copy(Vector& tmp, Vector& x)
{
}


template<>
void calc_row_sum<CSRMatrix>(CSRMatrix* A, double* x, int row_start, int row_end, double* row_sum, int row)
{
    for (int j = row_start; j < row_end; j++)
    {
        int col = A->idx2[j];
        if (col != row)
            *row_sum += A->vals[j] * x[col];
    }
}

template<>
void calc_row_sum<BSRMatrix>(BSRMatrix* A, double* x, int row_start, int row_end, double* row_sum, int row)
{
    char trans = 'T';
    int n = A->b_rows;
    int incr = 1;
    double one = 1.0;
    double** bdata = (double**)(A->get_data());

    // Block dot product between block row and vector x
    for (int j = row_start; j < row_end; j++)
    {
        int col = A->idx2[j];
        if (col != row)
            dgemv_(&trans, &n, &n, &one, bdata[j], &n, &((x[col * n])), &incr, &one, row_sum, &incr);
    }
}

template<typename MatrixType>
void update_row(MatrixType* A, double* x, double* b, double* tmp, double* diag, double* row_sum, double omega, double* tmp_rsum);

template<>
void update_row<CSRMatrix>(CSRMatrix* A, double* x, double* b, double* tmp, double* diag, double* row_sum, double omega, double* tmp_rsum)
{
    if (fabs(*diag) > zero_tol)
        *x = ((1.0 - omega) * *tmp) + (omega*((*b - *row_sum) / *diag));
}

template<>
void update_row<BSRMatrix>(BSRMatrix* A, double* x_row, double* b_row, double* tmp_row, double* D_inv_row, double* row_sum, double omega, double* tmp_rsum)
{
    char trans = 'T';
    int n = A->b_rows;
    int incr = 1;
    double one = 1.0;
    double zero = 0.0;

    // r = b - r / diag
    // in block form, calculate as: block_r = (b - block_r)*D_inv
    for (int k = 0; k < n; k++)
        row_sum[k] = b_row[k] - row_sum[k];
    dgemv_(&trans, &n, &n, &one, D_inv_row, &n, row_sum, &incr, &zero, tmp_rsum, &incr);
    
    // Weighted update of block of rows        
    for (int k = 0; k < A->b_rows; k++)
        x_row[k] = omega*tmp_rsum[k] + (1.0-omega)*tmp_row[k];
}

template <typename MatrixType>
void relax_row(MatrixType* A, Vector& b, Vector& x, Vector& tmp, double omega, 
        int row, double* rsum, double* tmp_rsum, double* D_inv);

template <>
void relax_row<CSRMatrix>(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, double omega, int row,
    double* rsum, double* tmp_rsum, double* D_inv)
{
    double diag = 0;
    double row_sum = 0;
    int row_start = A->idx1[row];
    int row_end = A->idx1[row+1];

    if (row_start < row_end && A->idx2[row_start] == row)
        diag = A->vals[row_start++];
    else return;

    calc_row_sum(A, tmp.data(), row_start, row_end, &row_sum, row);
    update_row(A, &(x[row]), &(b[row]), &(tmp[row]), &diag, &row_sum, omega, NULL);
}

template<>
void relax_row<BSRMatrix>(BSRMatrix* A, Vector& b, Vector& x, Vector& tmp, double omega, int row, 
        double* rsum, double* tmp_rsum, double* D_inv)
{
    int n = A->b_rows;

    int row_start = A->idx1[row];
    int row_end = A->idx1[row+1];
    if (row_start == row_end) 
        return;

    memset(rsum, 0, n*sizeof(double));

    // Block dot product between block row and vector x
    calc_row_sum(A, tmp.data(), row_start, row_end, rsum, row);

    update_row(A, &(x[row*n]), &(b[row*n]), &(tmp[row*n]), &(D_inv[row*(A->b_size)]),  
            rsum, omega, tmp_rsum);
}

template <typename MatrixType>
void relax_incr(MatrixType* A, Vector& b, Vector& x, Vector& tmp,
        double omega, double* D_inv = NULL, double* rsum = NULL, double* tmp_rsum = NULL,
        int* points = NULL, int points_len = 0)
{
    for (int row = 0; row < A->n_rows; row++)
    {
        relax_row(A, b, x, tmp, omega, row, rsum, tmp_rsum, D_inv);
    }
}

template <typename MatrixType>
void relax_decr(MatrixType* A, Vector& b, Vector& x, Vector& tmp,
        double omega, double* D_inv = NULL, double* rsum = NULL, double* tmp_rsum = NULL,
        int* points = NULL, int points_len = 0)
{
    for (int row = A->n_rows-1; row >= 0; row--)
    {
        relax_row(A, b, x, tmp, omega, row, rsum, tmp_rsum, D_inv);
    }
}

template <typename MatrixType>
void relax_points(MatrixType* A, Vector& b, Vector& x, Vector& tmp,
        double omega, double* D_inv = NULL, double* rsum = NULL, double* tmp_rsum = NULL,
        int* points = NULL, int points_len = 0)
{
    int idx = 0;
    while (idx < points_len)
    {
        int row = points[idx++];
        relax_row(A, b, x, tmp, omega, row, rsum, tmp_rsum, D_inv);
    }
}


template <typename C, typename R, typename M>
void relax(R relax_sweep, C copy, M* A, Vector& b, Vector& x, Vector& tmp,
        int num_sweeps, double omega, double* D_inv = NULL, int* points = NULL, int points_len = 0)
{
    A->sort();
    A->move_diag();

    double* rsum = new double[A->b_rows];
    double* tmp_rsum = new double[A->b_rows];

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        copy(tmp, x);
        relax_sweep(A, b, x, tmp, omega, D_inv, rsum, tmp_rsum, points, points_len);
    }

    delete[] rsum;
    delete[] tmp_rsum;
}

template <typename MatrixType>
void jacobi(MatrixType* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps, 
        double omega, double* D_inv, int* points, int points_len)
{
    auto F = relax_incr<MatrixType>;
    if (points_len > 0 && points != NULL)
        F = relax_points<MatrixType>;

    relax(F, jacobi_copy, A, b, x, tmp, num_sweeps, omega, D_inv, points, points_len);        
}

template <typename MatrixType>
void sor(MatrixType* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps,
        double omega, double* D_inv, int* points, int points_len)
{
    auto F = relax_incr<MatrixType>;
    if (points_len > 0 && points != NULL)
        F = relax_points<MatrixType>;

    relax(F, sor_copy, A, b, x, x, num_sweeps, omega, D_inv, points, points_len);
}

template void jacobi<CSRMatrix>(CSRMatrix*, Vector&, Vector&, Vector&, int, double, double*, int*, int);
template void jacobi<BSRMatrix>(BSRMatrix*, Vector&, Vector&, Vector&, int, double, double*, int*, int);

template void sor<CSRMatrix>(CSRMatrix*, Vector&, Vector&, Vector&, int, double, double*, int*, int);
template void sor<BSRMatrix>(BSRMatrix*, Vector&, Vector&, Vector&, int, double, double*, int*, int);



}
