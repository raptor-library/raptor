// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/types.hpp"
#include "interpolation.hpp"

using namespace raptor;

CSRMatrix* direct_interpolation(CSRMatrix* A,
        CSRBoolMatrix* S, const std::vector<int>& states)
{
    int start, end, col;
    int idx, new_idx, ctr;
    double sum_strong_pos, sum_strong_neg;
    double sum_all_pos, sum_all_neg;
    double val, alpha, beta, diag;
    double neg_coeff, pos_coeff;

    if (!A->diag_first)
    {
        A->move_diag();
    }
    if (!S->diag_first)
    {
        S->move_diag();
    }

    // Copy entries of A into sparsity pattern of S
    std::vector<double> sa;
    if (S->nnz)
    {
        sa.resize(S->nnz);
    }
    for (int i = 0; i < A->n_rows; i++)
    {
        start = S->idx1[i];
        end = S->idx1[i+1];
        if (S->idx2[start] == i) // skip over diag
        {
            start++;
        }
        ctr = A->idx1[i];
        for (int j = start; j < end; j++)
        {
            col = S->idx2[j];
            while (A->idx2[ctr] != col)
            {
                ctr++;
            }
            sa[j] = A->vals[ctr];
        }
    }

    std::vector<int> col_to_new;
    if (A->n_cols)
    {
        col_to_new.resize(A->n_cols, -1);
    }

    ctr = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        if (states[i])
        {
            col_to_new[i] = ctr++;
        }
    }

    CSRMatrix* P = new CSRMatrix(A->n_rows, ctr, A->nnz);

    P->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        if (states[i] == 1)
        {
            P->idx2.push_back(col_to_new[i]);
            P->vals.push_back(1);
        }
        else
        {
            sum_strong_pos = 0;
            sum_strong_neg = 0;
            sum_all_pos = 0;
            sum_all_neg = 0;

            start = S->idx1[i];
            end = S->idx1[i+1];
            if (S->idx2[start] == i)
            {
                start++;
            }
            for (int j = start; j < end; j++)
            {
                col = S->idx2[j];
                if (states[col] == 1)
                {
                    val = sa[j];
                    if (val < 0)
                    {
                        sum_strong_neg += val;
                    }
                    else
                    {
                        sum_strong_pos += val;
                    }
                }
            }

            start = A->idx1[i];
            end = A->idx1[i+1];
            diag = A->vals[start];
            for (int j = start+1; j < end; j++)
            {
                col = A->idx2[j];
                val = A->vals[j];
                if (val < 0)
                {
                    sum_all_neg += val;
                }
                else
                {
                    sum_all_pos += val;
                }
            }

            alpha = sum_all_neg / sum_strong_neg;

            if (sum_strong_pos == 0)
            {
                diag += sum_all_pos;
                beta = 0;
            }
            else
            {
                beta = sum_all_pos / sum_strong_pos;
            }

            neg_coeff = -alpha / diag;
            pos_coeff = -beta / diag;

            start = S->idx1[i];
            end = S->idx1[i+1];
            if (S->idx2[start] == i)
            {
                start++;
            }
            for (int j = start; j < end; j++)
            {
                col = S->idx2[j];
                if (states[col] == 1)
                {
                    val = sa[j];
                    P->idx2.push_back(col_to_new[col]);
                    if (val < 0)
                    {
                        P->vals.push_back(neg_coeff * val);
                    }
                    else
                    {
                        P->vals.push_back(pos_coeff * val);
                    }
                }
            }
        }
        P->idx1[i+1] = P->idx2.size();
    }
    P->nnz = P->idx2.size();

    return P;



}




