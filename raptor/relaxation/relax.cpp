// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "relaxation/relax.hpp"

using namespace raptor;

void jacobi(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps, 
        double omega)
{
    int row_start, row_end, vec_offset;
    double diag, row_sum;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        for (int i = 0; i < A->n_rows*b.b_vecs; i++)
        {
            tmp[i] = x[i];
        }

        for (int v = 0; v < b.b_vecs; v++)
        {
            vec_offset = v * A->n_rows;
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
                        row_sum += A->vals[j] * tmp[vec_offset + col];
                }
                if (fabs(diag) > zero_tol)
                    x[vec_offset + i] = ((1.0 - omega)*tmp[vec_offset + i]) + (omega*((b[vec_offset + i] - row_sum) / diag));
            }
        }
    }
}

void sor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps,
        double omega)
{
    int row_start, row_end, vec_offset;
    double diag_inv;
    double orig_x = 0;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        for (int v = 0; v < b.b_vecs; v++)
        {
            vec_offset = v * A->n_rows;
            for (int i = 0; i < A->n_rows; i++)
            {
                orig_x = x[vec_offset + i];
                x[vec_offset + i] = b[vec_offset + i];
                row_start = A->idx1[i];
                row_end = A->idx1[i+1];
                if (row_start == row_end) continue;

                diag_inv = omega / A->vals[row_start];
                for (int j = row_start + 1; j < row_end; j++)
                {
                    x[vec_offset + i] -= A->vals[j] * x[vec_offset + A->idx2[j]];
                }
                x[vec_offset + i] = diag_inv*x[vec_offset + i] + (1 - omega) * orig_x;
            }
        }
    }
}

void ssor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, int num_sweeps,
        double omega)
{
    int row_start, row_end, vec_offset;
    double diag_inv;
    double orig_x = 0;

    for (int iter = 0; iter < num_sweeps; iter++)
    {
        for (int v = 0; v < b.b_vecs; v++)
        {
            vec_offset = v * A->n_rows;
            for (int i = 0; i < A->n_rows; i++)
            {
                orig_x = x[vec_offset + i];
                x[vec_offset + i] = b[vec_offset + i];
                row_start = A->idx1[i];
                row_end = A->idx1[i+1];
                if (row_start == row_end) continue;

                diag_inv = omega / A->vals[row_start];
                for (int j = row_start + 1; j < row_end; j++)
                {
                    x[vec_offset + i] -= A->vals[j] * x[vec_offset + A->idx2[j]];
                }
                x[vec_offset + i] = diag_inv*x[vec_offset + i] + (1 - omega) * orig_x;
            }

            for (int i = A->n_rows - 1; i >= 0; i--)
            {
                orig_x = x[vec_offset + i];            
                x[vec_offset + i] = b[vec_offset + i];
                row_start = A->idx1[i];
                row_end = A->idx1[i+1];
                if (row_start == row_end) continue;

                diag_inv = omega / A->vals[row_start];
                for (int j = row_start + 1; j < row_end; j++)
                {
                    x[vec_offset + i] -= A->vals[j] * x[vec_offset + A->idx2[j]];
                }
                x[vec_offset + i] = diag_inv*x[vec_offset + i] + (1 - omega) * orig_x;
            }
        }
    }
}



