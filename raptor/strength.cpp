// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "core/matrix.hpp"

using namespace raptor;

CSRMatrix* classical_strength(CSRMatrix* A, double theta, int num_variables, int* variables)
{
    int start, end, col;
    double val;
    double row_scale;
    double threshold;
    double diag;

    if (!A->sorted)
    {
        A->sort();
    }
    if (!A->diag_first)
    {
        A->move_diag();
    }

    CSRMatrix* S = new CSRMatrix(A->n_rows, A->n_cols, A->nnz);

    S->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        // Always add the diagonal 
        start = A->idx1[i];
        end = A->idx1[i+1];
        if (end - start)
        {
            if (A->idx2[start] == i)
            {
                diag = A->vals[start];
                start++;
            }
            else
            {
                diag = 0.0;
            }

            if (num_variables == 1)
            {
                if (diag < 0.0) // find max off-diag value in row
                {
                    row_scale = -RAND_MAX;
                    for (int j = start; j < end; j++)
                    {
                        val = A->vals[j];
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
                        val = A->vals[j];
                        if (val < row_scale)
                        {
                            row_scale = val;
                        }
                    }
                }
            }
            else
            {
                if (diag < 0.0) // find max off-diag value in row
                {
                    row_scale = -RAND_MAX;
                    for (int j = start; j < end; j++)
                    {
                        col = A->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->vals[j];
                            if (val > row_scale)
                            {
                                row_scale = val;
                            }
                        }
                    }
                }
                else // find min off-diag value in row
                {
                    row_scale = RAND_MAX;
                    for (int j = start; j < end; j++)
                    {
                        col = A->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->vals[j];
                            if (val < row_scale)
                            {
                                row_scale = val;
                            }
                        }
                    }
                }

            }

            // Multiply row magnitude by theta
            threshold = row_scale*theta;

            // Always add diagonal
            S->idx2.push_back(i);
            S->vals.push_back(diag);

            // Add off-diagonals greater than threshold
            if (num_variables == 1)
            {
                if (diag < 0)
                {
                    for (int j = start; j < end; j++)
                    {
                        val = A->vals[j];
                        if (val > threshold)
                        {
                            S->idx2.push_back(A->idx2[j]);
                            S->vals.push_back(val);
                        }
                    }
                }
                else
                {
                    for (int j = start; j < end; j++)
                    {
                        val = A->vals[j];
                        if (val < threshold)
                        {
                            S->idx2.push_back(A->idx2[j]);
                            S->vals.push_back(val);
                        }
                    }
                }
            }
            else
            {
                if (diag < 0)
                {
                    for (int j = start; j < end; j++)
                    {
                        col = A->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->vals[j];
                            if (val > threshold)
                            {
                                S->idx2.push_back(col);
                                S->vals.push_back(val);
                            }
                        }
                    }
                }
                else
                {
                    for (int j = start; j < end; j++)
                    {
                        col = A->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->vals[j];
                            if (val < threshold)
                            {
                                S->idx2.push_back(col);
                                S->vals.push_back(val);
                            }
                        }
                    }
                }
            }
        }
        S->idx1[i+1] = S->idx2.size();
    }
    S->nnz = S->idx2.size();

    return S;

}

CSRMatrix* symmetric_strength(CSRMatrix* A, double theta)
{
    int start, end, col;
    double val;
    double eps;
    aligned_vector<double> diags(A->n_rows, 0);

    if (!A->sorted)
    {
        A->sort();
    }
    if (!A->diag_first)
    {
        A->move_diag();
    }

    CSRMatrix* S = new CSRMatrix(A->n_rows, A->n_cols, A->nnz);

    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        if (end - start && A->idx2[start] == i)
            diags[i] = fabs(A->vals[start]);
    }

    S->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        eps = theta * theta * diags[i];

        // Always add the diagonal 
        start = A->idx1[i];
        end = A->idx1[i+1];

        if (end - start && A->idx2[start] == i)
        {
            S->vals.push_back(A->vals[start]);
            start++;
        }
        else
        {
            S->vals.push_back(0.0);
        }
        S->idx2.push_back(i);
             
        // Add off-diagonals greater than threshold
        for (int j = start; j < end; j++)
        {
            col = A->idx2[j];
            val = A->vals[j];
            if (val*val >= eps * diags[col])
            {
                S->idx2.push_back(col);
                S->vals.push_back(val);
            }
        }
        S->idx1[i+1] = S->idx2.size();
    }
    S->nnz = S->idx2.size();

    return S;

}

CSRMatrix* CSRMatrix::strength(strength_t strength_type,
        double theta, int num_variables, int* variables)
{
    switch (strength_type)
    {
        case Classical:
            return classical_strength(this, theta, num_variables, variables);
        case Symmetric:
            return symmetric_strength(this, theta);
    }
}

