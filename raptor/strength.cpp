#include "core/par_matrix.hpp"

using namespace raptor;

// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)
CSRMatrix* CSRMatrix::strength(double theta)
{
    int start, end;
    double val, abs_val;
    double row_max;
    double threshold;

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
            // Find max off-diagonal
            row_max = 0.0;
            for (int j = start + 1; j < end; j++)
            {
                abs_val = fabs(vals[j]);
                if (abs_val > row_max)
                {
                    row_max = abs_val;
                }
            }
            threshold = row_max*theta;

            // Find largest magnitude in row (including diag)
            abs_val = fabs(vals[start]);
            if (abs_val > row_max)
            {
                row_max = abs_val;
            }

            // For each value to be added... add absolute value of 
            // Aij / row_max
            
            // Always add diagonal
            S->idx2.push_back(i);
            S->vals.push_back(fabs(vals[start]) / row_max);


            // Add off-diagonals greater than threshold
            for (int j = start + 1; j < end; j++)
            {
                val = vals[j];
                if (fabs(val) >= threshold)
                {
                    S->idx2.push_back(idx2[j]);
                    S->vals.push_back(fabs(val) / row_max);
                }
            }
        }
        S->idx1[i+1] = S->idx2.size();
    }
    S->nnz = S->idx2.size();

    return S;

}

