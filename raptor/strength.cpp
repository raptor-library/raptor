// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/matrix.hpp"

using namespace raptor;

// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)
CSRMatrix* CSRMatrix::strength(double theta)
{
    int start, end;
    double val;
    double row_scale;
    double threshold;
    double diag;

    if (!sorted)
    {
        sort();
    }
    if (!diag_first)
    {
        move_diag();
    }

    CSRMatrix* S = new CSRMatrix(n_rows, n_cols, nnz);

    S->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        // Always add the diagonal 
        start = idx1[i];
        end = idx1[i+1];
        if (end - start)
        {
            if (idx2[start] == i)
            {
                diag = vals[start];
                start++;
            }
            else
            {
                diag = 0.0;
            }

            if (diag < 0.0) // find max off-diag value in row
            {
                row_scale = -RAND_MAX;
                for (int j = start; j < end; j++)
                {
                    val = vals[j];
                    if (val > row_scale)
                    {
                        row_scale = val;
                    }
                }
            }
            else // find min off-diag value in row
            {
                row_scale = RAND_MAX;
                for (int j = start; j < end; j++)
                {
                    val = vals[j];
                    if (val < row_scale)
                    {
                        row_scale = val;
                    }
                }
            }

            // Multiply row magnitude by theta
            threshold = row_scale*theta;

            // Always add diagonal
            S->idx2.push_back(i);
            S->vals.push_back(diag);

            // Add off-diagonals greater than threshold
            if (diag < 0)
            {
                for (int j = start; j < end; j++)
                {
                    val = vals[j];
                    if (val > threshold)
                    {
                        S->idx2.push_back(idx2[j]);
                        S->vals.push_back(vals[j]);
                    }
                }
            }
            else
            {
                for (int j = start; j < end; j++)
                {
                    val = vals[j];
                    if (val < threshold)
                    {
                        S->idx2.push_back(idx2[j]);
                        S->vals.push_back(vals[j]);
                    }
                }
            }
        }
        S->idx1[i+1] = S->idx2.size();
    }
    S->nnz = S->idx2.size();

    return S;

}

