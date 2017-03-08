// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

void CSCMatrix::jacobi(Vector& x, Vector& b, Vector& tmp, double omega)
{
    int col_start, col_end, row;
    std::vector<double> diags(n_rows, 0);

    // Tmp[i] will gather row sum
    for (int i = 0; i < n_rows; i++)
    {
        tmp[i] = 0;
    }

    // Find diagonal values and row_sums
    for (int i = 0; i < n_cols; i++)
    {
        col_start = idx1[i];
        col_end = idx1[i+1];
        if (col_end - col_start)
            diags[i] = vals[col_start];
        for (int j = col_start+1; j < col_end; j++)
        {
            row = idx2[j];
            tmp[row] += vals[j] * x[i];
        }
    }

    // Update x
    for (int i = 0; i < n_rows; i++)
    {
        if (fabs(diags[i]) > zero_tol)
            x[i] = ((1.0 - omega)*x[i]) + (omega*((b[i] - tmp[i]) / diags[i]));
    }
}

void Matrix::jacobi(Vector& x, Vector& b, Vector& tmp, double omega)
{
    if (format() == COO)
    {
        printf("This matrix format is not supported.\n");
        return;
    }

    int row_start, row_end;
    double diag, row_sum;

    for (int i = 0; i < n_rows; i++)
    {
        tmp[i] = x[i];
    }

    for (int i = 0; i < n_rows; i++)
    {
        row_start = idx1[i];
        row_end = idx1[i+1];
        if (row_start == row_end) continue;
        row_sum = 0;
        diag = 0;

        for (int j = row_start; j < row_end; j++)
        {
            int col = idx2[j];
            if (i == col)
                diag = vals[j];
            else
                row_sum += vals[j] * tmp[col];
        }
        if (fabs(diag) > zero_tol)
            x[i] = ((1.0 - omega)*tmp[i]) + (omega*((b[i] - row_sum) / diag));
    }
}

void Matrix::gauss_seidel(Vector& x, Vector& b)
{
    if (format() == COO || format() == CSC)
    {
        printf("This matrix format is not supported.\n");
        return;
    }

    int row_start, row_end;
    double diag_inv;

    for (int i = 0; i < n_rows; i++)
    {
        x[i] = b[i];
        row_start = idx1[i];
        row_end = idx1[i+1];
        if (row_start == row_end) continue;

        diag_inv = 1.0 / vals[row_start];
        for (int j = row_start + 1; j < row_end; j++)
        {
            x[i] -= vals[j] * x[idx2[j]];
        }
        x[i] *= diag_inv;
    }
}

void Matrix::SOR(Vector& x, Vector& b, double omega)
{
    if (format() == COO || format() == CSC)
    {
        printf("This matrix format is not supported.\n");
        return;
    }

    int row_start, row_end;
    double diag_inv;
    double orig_x = 0;

    for (int i = 0; i < n_rows; i++)
    {
        orig_x = x[i];
        x[i] = b[i];
        row_start = idx1[i];
        row_end = idx1[i+1];
        if (row_start == row_end) continue;

        diag_inv = omega / vals[row_start];
        for (int j = row_start + 1; j < row_end; j++)
        {
            x[i] -= vals[j] * x[idx2[j]];
        }
        x[i] = diag_inv*x[i] + (1 - omega) * orig_x;
    }
}


