// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

void Matrix::classical_strength(CSRMatrix* S, double theta)
{
    printf("This matrix format is not supported.\n");
}

void Matrix::symmetric_strength(CSCMatrix* S, double theta)
{
    printf("This matrix format is not supported.\n");
}
void Matrix::symmetric_strength(CSRMatrix* S, double theta)
{
    printf("This matrix format is not supported.\n");
}

void CSRMatrix::classical_strength(CSRMatrix* S, double theta)
{
    S->n_rows = n_rows;
    S->n_cols = n_cols;
    S->idx1.resize(n_rows + 1);
    S->idx2.resize(nnz);
    S->vals.resize(nnz);

    int ctr = 0;
    S->idx1[0] = ctr;

    int row_start, row_end;
    double val, max_val;

    for (int i = 0; i < n_rows; i++)
    {
        row_start = idx1[i];
        row_end = idx1[i+1];

        // Find maximum off-diagonal value in row
        max_val = 0;
        for (int j = row_start + 1; j < row_end; j++)
        {
            val = fabs(vals[j]);
            if (val > max_val)
                max_val = val;
        }

        // Always add the diagonal
        if (row_end - row_start)
        {
            S->idx2[ctr] = idx2[row_start];
            S->vals[ctr] = 1.0;
            ctr++;
        }

        // Add off-diagonal values if
        // |Aij| >= theta * max(|Aik|) for k != i
        for (int j = row_start+1; j < row_end; j++)
        {
            if (fabs(vals[j]) >= theta * max_val)
            {
                S->idx2[ctr] = idx2[j];
                S->vals[ctr] = 1.0;
                ctr++;
            }
        }

        S->idx1[i+1] = ctr;
    }
    S->nnz = ctr;
}

void CSCMatrix::symmetric_strength(CSCMatrix* S, double theta)
{
    S->n_rows = n_rows;
    S->n_cols = n_cols;
    S->idx1.resize(n_cols + 1);
    S->idx2.resize(nnz);
    S->vals.resize(nnz);

    int col_start, col_end;
    int row;
    double eps_diag;
    double theta_sq = theta*theta;
    double val, val_sq;

    std::vector<double> abs_diag(n_cols);
    for (int i = 0; i < n_cols; i++)
    {
        col_start = idx1[i];
        if (idx1[i+1] - col_start)
        {
            abs_diag[i] = fabs(vals[col_start]);
        }
        else
        {
            abs_diag[i] = 0.0;
        }
    }

    int ctr = 0;
    S->idx1[0] = ctr;
    for (int i = 0; i < n_cols; i++)
    {
        col_start = idx1[i];
        col_end = idx1[i+1];

        if (col_end - col_start)
        {
            eps_diag = abs_diag[i] * theta_sq;

            S->idx2[ctr] = idx2[col_start];
            S->vals[ctr] = 1.0;
            ctr++;

            for (int j = col_start+1; j < col_end; j++)
            {
                row = idx2[j];
                val = vals[j];
                val_sq = val*val;

                if (val_sq >= eps_diag * abs_diag[row])
                {
                    S->idx2[ctr] = row;
                    S->vals[ctr] = val;
                    ctr++;
                }
            }
        }
        S->idx1[i+1] = ctr;
    }
    S->nnz = ctr;
}

void CSRMatrix::symmetric_strength(CSRMatrix* S, double theta)
{
    S->n_rows = n_rows;
    S->n_cols = n_cols;
    S->idx1.resize(n_rows + 1);
    S->idx2.resize(nnz);
    S->vals.resize(nnz);

    int row_start, row_end;
    int col;
    double eps_diag;
    double theta_sq = theta*theta;
    double val, val_sq;

    std::vector<double> abs_diag(n_rows);
    for (int i = 0; i < n_rows; i++)
    {
        row_start = idx1[i];
        if (idx1[i+1] - row_start)
        {
            abs_diag[i] = fabs(vals[row_start]);
        }
        else
        {
            abs_diag[i] = 0.0;
        }
    }

    int ctr = 0;
    S->idx1[0] = ctr;
    for (int i = 0; i < n_rows; i++)
    {
        row_start = idx1[i];
        row_end = idx1[i+1];

        if (row_end - row_start)
        {
            eps_diag = abs_diag[i] * theta_sq;

            S->idx2[ctr] = idx2[row_start];
            S->vals[ctr] = 1.0;
            ctr++;

            for (int j = row_start+1; j < row_end; j++)
            {
                col = idx2[j];
                val = vals[j];
                val_sq = val*val;

                if (val_sq >= eps_diag * abs_diag[col])
                {
                    S->idx2[ctr] = col;
                    S->vals[ctr] = val;
                    ctr++;
                }
            }
        }
        S->idx1[i+1] = ctr;
    }
    S->nnz = ctr;
}



