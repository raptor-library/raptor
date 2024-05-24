// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "raptor/core/vector.hpp"
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

void jacobi(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps, 
        double omega)
{
    int row_start, row_end;
    double diag, row_sum;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            tmp[i] = x[i];
        }

        for (int i = 0; i < A->n_rows; i++)
        {
            row_start = A->idx1[i];
            row_end = A->idx1[i+1];
            if (row_start == row_end) continue;
            row_sum = 0;
            diag = 0;

            for (int j = row_start; j < row_end; j++)
            {
                int col = A->idx2[j];
                if (i == col)
                    diag = A->vals[j];
                else
                    row_sum += A->vals[j] * tmp[col];
            }
            if (fabs(diag) > zero_tol)
                x[i] = ((1.0 - omega)*tmp[i]) + (omega*((b[i] - row_sum) / diag));
        }
    }
}

void sor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps,
        double omega)
{
    int row_start, row_end;
    double diag;
    double rsum;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            rsum = 0;
            diag = 0;
            row_start = A->idx1[i];
            row_end = A->idx1[i+1];

            for (int j = row_start; j < row_end; j++)
            {
                int col = A->idx2[j];
                if (i == col)
                    diag = A->vals[j];
                else
                    rsum += A->vals[j] * x[col];
            }
            
            if (fabs(diag) > zero_tol)
                x[i] = omega*(b[i] - rsum)/diag + (1 - omega) * x[i];
        //        x[i] = (b[i] - rsum) / diag;
        }
    }
}

void ssor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps,
        double omega)
{
    int row_start, row_end;
    double diag_inv;
    double orig_x = 0;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            orig_x = x[i];
            x[i] = b[i];
            row_start = A->idx1[i];
            row_end = A->idx1[i+1];
            if (row_start == row_end) continue;

            diag_inv = omega / A->vals[row_start];
            for (int j = row_start + 1; j < row_end; j++)
            {
                x[i] -= A->vals[j] * x[A->idx2[j]];
            }
            x[i] = diag_inv*x[i] + (1 - omega) * orig_x;
        }

        for (int i = A->n_rows - 1; i >= 0; i--)
        {
            orig_x = x[i];            
            x[i] = b[i];
            row_start = A->idx1[i];
            row_end = A->idx1[i+1];
            if (row_start == row_end) continue;

            diag_inv = omega / A->vals[row_start];
            for (int j = row_start + 1; j < row_end; j++)
            {
                x[i] -= A->vals[j] * x[A->idx2[j]];
            }
            x[i] = diag_inv*x[i] + (1 - omega) * orig_x;
        }
    }
}


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


// From Pyamg block_jacobi https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/amg_core/relaxation.h#L914
void jacobi(BSRMatrix* A, double* D_inv, Vector& b, Vector& x, Vector& tmp, int num_sweeps, 
       double omega)
{
    double* rsum = new double[A->b_rows];
    double* tmp_rsum = new double[A->b_rows];

    double** bdata = (double**)(A->get_data());
    int row_start, row_end;
    char trans = 'T';
    int n = A->b_rows;
    int incr = 1;
    double one = 1.0;
    double zero = 0.0;

    // Go through all sweeps
    for (int iter = 0; iter < num_sweeps; iter++)
    {
        // Copy x to tmp vector
        memcpy(tmp.data(), x.data(), tmp.size()*sizeof(double));

        // Begin block Jacobi sweep
        for (int row = 0; row < A->n_rows; row++)
        {
            row_start = A->idx1[row];
            row_end = A->idx1[row+1];
            if (row_start == row_end) continue;

            memset(rsum, 0, n*sizeof(double));

            // Block dot product between block row and vector x
            for (int j = row_start; j < row_end; j++)
            {
                int col = A->idx2[j];
                if (row != col) //ignore diagonal
                {
                    dgemv_(&trans, &n, &n, &one, bdata[j], &n, &((tmp[col * n])), &incr, &one, rsum, &incr);
                }
            }

            // r = b - r / diag
            // in block form, calculate as: block_r = (b - block_r)*D_inv
            for (int k = 0; k < n; k++)
                rsum[k] = b[row*n + k] - rsum[k];
            dgemv_(&trans, &n, &n, &one, &(D_inv[row*A->b_size]), &n, rsum, &incr, &zero, tmp_rsum, &incr);
            
            // Weighted Jacobi calculation for row
            for (int k = 0; k < A->b_rows; k++)
                x[row*n + k] = omega*tmp_rsum[k] + (1.0-omega)*tmp[row*n + k];

        }
    } 

    delete[] rsum;
    delete[] tmp_rsum;
}

void sor(BSRMatrix* A, double* D_inv, Vector& b, Vector& x, Vector& tmp, int num_sweeps, 
       double omega)
{
    double* rsum = new double[A->b_rows];
    double* tmp_rsum = new double[A->b_rows];

    double** bdata = (double**)(A->get_data());
    int row_start, row_end;
    char trans = 'T';
    int n = A->b_rows;
    int incr = 1;
    double one = 1.0;
    double zero = 0.0;

    // Go through all sweeps
    for (int iter = 0; iter < num_sweeps; iter++)
    {
        // Begin block Jacobi sweep
        for (int row = 0; row < A->n_rows; row++)
        {
            row_start = A->idx1[row];
            row_end = A->idx1[row+1];
            if (row_start == row_end) continue;

            memset(rsum, 0, n*sizeof(double));

            // Block dot product between block row and vector x
            for (int j = row_start; j < row_end; j++)
            {
                int col = A->idx2[j];
                if (row != col) //ignore diagonal
                {
                    dgemv_(&trans, &n, &n, &one, bdata[j], &n, &((x[col * n])), &incr, &one, rsum, &incr);
                }
            }

            // r = b - r / diag
            // in block form, calculate as: block_r = (b - block_r)*D_inv
            for (int k = 0; k < n; k++)
                rsum[k] = b[row*n + k] - rsum[k];
            dgemv_(&trans, &n, &n, &one, &(D_inv[row*A->b_size]), &n, rsum, &incr, &zero, tmp_rsum, &incr);
            
            // Weighted Jacobi calculation for row
            for (int k = 0; k < A->b_rows; k++)
                x[row*n + k] = omega*tmp_rsum[k] + (1.0-omega)*x[row*n + k];

        }
    } 

    delete[] rsum;
    delete[] tmp_rsum;
}

void block_relax_free(double* A_inv)
{
    delete[] A_inv;
}


}
