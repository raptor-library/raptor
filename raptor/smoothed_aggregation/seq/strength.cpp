// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

Matrix* Matrix::strength(double theta)
{
    printf("This matrix format is not supported.\n");
    return NULL;
}

CSRMatrix* CSRMatrix::strength(double theta)
{
    CSRMatrix* S = new CSRMatrix(n_rows, n_cols);
    S->idx2.reserve(nnz);
    S->vals.reserve(nnz);

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

    S->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        row_start = idx1[i];
        row_end = idx1[i+1];

        if (row_end - row_start)
        {
            eps_diag = abs_diag[i] * theta_sq;

            S->idx2.push_back(idx2[row_start]);
            S->vals.push_back(1.0);

            for (int j = row_start+1; j < row_end; j++)
            {
                col = idx2[j];
                val = vals[j];
                val_sq = val*val;

                if (val_sq >= eps_diag * abs_diag[col])
                {
                    S->idx2.push_back(col);
                    S->vals.push_back(val);
                }
            }
        }
        S->idx1[i+1] = S->idx2.size();
    }
    S->nnz = S->idx2.size();

    return S;
}



