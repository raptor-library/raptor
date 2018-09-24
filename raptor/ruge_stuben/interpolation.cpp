// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "assert.h"
#include "core/types.hpp"
#include "interpolation.hpp"

using namespace raptor;

CSRMatrix* extended_interpolation(CSRMatrix* A, CSRMatrix* S, 
        const aligned_vector<int>& states, int num_variables, int* variables)
{
    int startA, endA;
    int startS, endS;
    int startNS, endNS;
    int startSS, endSS;
    int startSU, endSU;
    int startNU, endNU;
    int start_k, end_k, col_k;
    int col, ctr, idx;
    int row_start, row_end;
    int head, length;
    double weak_sum;
    double val;
    double sign;
    aligned_vector<int> pos;
    aligned_vector<int> row_coarse;
    aligned_vector<double> row_strong;
    aligned_vector<int> next;
    aligned_vector<double> weak_sums;
    aligned_vector<int> signs;
    aligned_vector<double> coarse_sum;
    if (A->n_rows)
    {
        pos.resize(A->n_rows, -1);
        row_coarse.resize(A->n_rows, 0);
        row_strong.resize(A->n_rows, 0);
        weak_sums.resize(A->n_rows, 0);
        signs.resize(A->n_rows, 1);
        coarse_sum.resize(A->n_rows, 0);
    }

    A->sort();
    A->move_diag();
    S->sort();
    S->move_diag();

    // Split matrices into the following:
    //      - NS: values in A but not S
    //      - SS: selected values in S
    //      - SU: unselected values in S
    CSRMatrix* NS = new CSRMatrix(A->n_rows, A->n_cols);
    CSRMatrix* SS = new CSRMatrix(A->n_rows, A->n_cols);
    CSRMatrix* SU = new CSRMatrix(A->n_rows, A->n_cols);
    CSRMatrix* NU = new CSRMatrix(A->n_rows, A->n_cols);
    NS->idx1[0] = 0;
    SS->idx1[0] = 0;
    SU->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        ctr = S->idx1[i] + 1;
        endS = S->idx1[i+1];
        startA = A->idx1[i];
        endA = A->idx1[i+1];
        // Add diagonal to NS
        weak_sums[i] = A->vals[startA++];
        if (weak_sums[i] < 0) 
            signs[i] = -1;
        for (int j = startA; j < endA; j++)
        {
            col = A->idx2[j];
            val = A->vals[j];
            if (ctr < endS && S->idx2[ctr] == col)
            {
                if (states[col] == Selected)
                {
                    SS->idx2.emplace_back(col);
                    SS->vals.emplace_back(val);
                }
                else if (states[col] == Unselected)
                {
                    SU->idx2.emplace_back(col);
                    SU->vals.emplace_back(val);
                    NU->idx2.emplace_back(col);
                    NU->vals.emplace_back(val);
                }
                ctr++;
            }
            else
            {
                if (num_variables == 1 || variables[i] == variables[col])
                {
                    weak_sums[i] += val;
                }

                if (states[col] == Selected)
                {
                    NS->idx2.emplace_back(col);
                    NS->vals.emplace_back(val);
                }
                else
                {
                    NU->idx2.emplace_back(col);
                    NU->vals.emplace_back(val);
                }
            }
        }
        NS->idx1[i+1] = NS->idx2.size();
        SS->idx1[i+1] = SS->idx2.size();
        SU->idx1[i+1] = SU->idx2.size();
        NU->idx1[i+1] = NU->idx2.size();
    }
    NU->nnz = NU->idx2.size();
    
    aligned_vector<int> col_to_new;
    if (A->n_cols)
    {
        col_to_new.resize(A->n_cols, -1);
    }

    ctr = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        if (states[i] == Selected)
        {
            col_to_new[i] = ctr++;
        }
    }

    // Form Sparsity Pattern of P
    CSRMatrix* P = new CSRMatrix(A->n_rows, ctr, A->nnz);

    CSCMatrix* U_csc = NU->to_CSC();
    delete NU;

    // Main loop.. add entries to P
    P->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        if (states[i] == Selected)
        {
            P->idx2.emplace_back(i);
            P->vals.emplace_back(1.0);
            P->idx1[i+1] = P->idx2.size();
            continue;
        }

        row_start = P->idx2.size();
        row_coarse[i] = 1;

        // Skip over diagonal values
        startNS = NS->idx1[i];
        endNS = NS->idx1[i+1];
        startSS = SS->idx1[i];
        endSS = SS->idx1[i+1];
        startSU = SU->idx1[i];
        endSU = SU->idx1[i+1];

        // Add strongly connected C points to P
        for (int j = startSS; j < endSS; j++)
        {
            col = SS->idx2[j];
            val = SS->vals[j];
            pos[col] = P->idx2.size();
            P->idx2.emplace_back(col);
            P->vals.emplace_back(val);
            row_coarse[col] = 1;
        }

        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            val = SU->vals[j];
            row_strong[col] = val;
        }

        sign = signs[i];
        weak_sum = weak_sums[i];

        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            // Add distance-2
            start_k = SS->idx1[col];
            end_k = SS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = SS->idx2[k];
                if (row_coarse[col_k] == 0)
                {
                    pos[col_k] = P->idx2.size();
                    P->idx2.emplace_back(col_k);
                    P->vals.emplace_back(0); // Aij is 0
                    row_coarse[col_k] = 1;
                } 
            }
        }
        row_end = P->idx2.size();


        // Find row coarse sums 
        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            coarse_sum[col] = 0;
            start_k = NS->idx1[col];
            end_k = NS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = NS->idx2[k];
                val = NS->vals[k] * row_coarse[col_k];
                if (val * sign < 0)
                {
                    coarse_sum[col] += val;
                }
            }
            start_k = SS->idx1[col];
            end_k = SS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = SS->idx2[k];
                val = SS->vals[k] * row_coarse[col_k];
                if (val * sign < 0)
                {
                    coarse_sum[col] += val;
                }
            }
        }

        // Add column i 
        startNU = U_csc->idx1[i];
        endNU = U_csc->idx1[i+1];
        for (int j = startNU; j < endNU; j++)
        {
            int row = U_csc->idx2[j];
            val = U_csc->vals[j];
            if (val * sign < 0)
                coarse_sum[row] += val;
        }

        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            if (fabs(coarse_sum[col]) < zero_tol)
            {
                weak_sum += SU->vals[j];
                row_strong[col] = 0;
            }
            else
            {
                row_strong[col] /= coarse_sum[col];
            }
        }

        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            start_k = NS->idx1[col];
            end_k = NS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = NS->idx2[k];
                val = NS->vals[k];
                idx = pos[col_k];
                if (val * sign < 0 && idx >= 0)
                {
                    P->vals[idx] += (row_strong[col] * val);
                }
            }
            start_k = SS->idx1[col];
            end_k = SS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = SS->idx2[k];
                val = SS->vals[k];
                idx = pos[col_k];
                if (val * sign < 0 && idx >= 0)
                {
                    P->vals[idx] += (row_strong[col] * val);
                }
            }
        }

        // Go through column i and add to weak sum
        startNU = U_csc->idx1[i];
        endNU = U_csc->idx1[i+1];
        for (int j = startNU; j < endNU; j++)
        {
            int row = U_csc->idx2[j];
            val = U_csc->vals[j];
            weak_sum += (row_strong[row] * val);
        }

        for (int j = row_start; j < row_end; j++)
        {
            P->vals[j] /= -weak_sum;
            col = P->idx2[j];
            row_coarse[col] = 0;
            row_strong[col] = 0;
            pos[col] = -1;
        }

        row_coarse[i] = 0;

        for (int i = 0; i < A->n_rows; i++)
            row_strong[i] = 0;

        P->idx1[i+1] = P->idx2.size();

    }
    P->nnz = P->idx2.size();

    for (aligned_vector<int>::iterator it = P->idx2.begin(); it != P->idx2.end(); ++it)
    {
        *it = col_to_new[*it];
    }

    delete NS;
    delete SS;
    delete SU;
    delete U_csc;

    return P;
}

CSRMatrix* mod_classical_interpolation(CSRMatrix* A, CSRMatrix* S, 
        const aligned_vector<int>& states, int num_variables, int* variables)
{
    int startA, endA;
    int startS, endS;
    int startNS, endNS;
    int startSS, endSS;
    int startSU, endSU;
    int startNU, endNU;
    int start_k, end_k, col_k;
    int col, ctr, idx;
    double weak_sum;
    double val;
    double sign;
    double coarse_sum;
    aligned_vector<int> pos;
    aligned_vector<int> row_coarse;
    aligned_vector<double> row_strong;
    aligned_vector<double> weak_sums;
    aligned_vector<int> signs;
    if (A->n_rows)
    {
        pos.resize(A->n_rows, -1);
        row_coarse.resize(A->n_rows, 0);
        row_strong.resize(A->n_rows, 0);
        weak_sums.resize(A->n_rows, 0);
        signs.resize(A->n_rows, 1);
    }

    A->sort();
    A->move_diag();
    S->sort();
    S->move_diag();

    // Split matrices into the following:
    //      - NS: values in A but not S
    //      - SS: selected values in S
    //      - SU: unselected values in S
    CSRMatrix* NS = new CSRMatrix(A->n_rows, A->n_cols);
    CSRMatrix* SS = new CSRMatrix(A->n_rows, A->n_cols);
    CSRMatrix* SU = new CSRMatrix(A->n_rows, A->n_cols);
    NS->idx1[0] = 0;
    SS->idx1[0] = 0;
    SU->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        ctr = S->idx1[i] + 1;
        endS = S->idx1[i+1];
        startA = A->idx1[i];
        endA = A->idx1[i+1];
        // Add diagonal to NS
        weak_sums[i] += A->vals[startA++];
        if (weak_sums[i] < 0)
            signs[i] = -1;
        for (int j = startA; j < endA; j++)
        {
            col = A->idx2[j];
            val = A->vals[j];
            if (ctr < endS && S->idx2[ctr] == col)
            {
                if (states[col] == Selected)
                {
                    SS->idx2.emplace_back(col);
                    SS->vals.emplace_back(val);
                }
                else if (states[col] == Unselected)
                {
                    SU->idx2.emplace_back(col);
                    SU->vals.emplace_back(val);
                }
                ctr++;
            }
            else
            {
                if (states[col] == Selected)
                {
                    NS->idx2.emplace_back(col);
                    NS->vals.emplace_back(val);
                }
                if (num_variables == 1 || variables[i] == variables[col])
                {
                    weak_sums[i] += val;
                }
            }
        }
        NS->idx1[i+1] = NS->idx2.size();
        SS->idx1[i+1] = SS->idx2.size();
        SU->idx1[i+1] = SU->idx2.size();
    }

    aligned_vector<int> col_to_new;
    if (A->n_cols)
    {
        col_to_new.resize(A->n_cols, -1);
    }

    ctr = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        if (states[i] == Selected)
        {
            col_to_new[i] = ctr++;
        }
    }

    // Form P
    CSRMatrix* P = new CSRMatrix(A->n_rows, ctr, A->nnz);

    // Main loop.. add entries to P
    P->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        if (states[i] == Selected)
        {
            P->idx2.emplace_back(col_to_new[i]);
            P->vals.emplace_back(1.0);
            P->idx1[i+1] = P->idx2.size();
            continue;
        }

        startSS = SS->idx1[i];
        endSS = SS->idx1[i+1];
        for (int j = startSS; j < endSS; j++)
        {
            col = SS->idx2[j];
            val = SS->vals[j];
            pos[col] = P->idx2.size();
            P->idx2.emplace_back(col_to_new[col]);
            P->vals.emplace_back(val);
            row_coarse[col] = 1;
        }

        // Add strong fine values to row_strong
        startSU = SU->idx1[i];
        endSU = SU->idx1[i+1];
        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            val = SU->vals[j];
            row_strong[col] = val;
        }

        // Add weak values to weak_sum
        startNS = NS->idx1[i];
        endNS = NS->idx1[i+1];

        weak_sum = weak_sums[i];
        sign = signs[i];

        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            coarse_sum = 0;
            start_k = SS->idx1[col];
            end_k = SS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = SS->idx2[k];
                val = SS->vals[k] * row_coarse[col_k];
                if (val * sign < 0)
                    coarse_sum += val;
            }
            start_k = NS->idx1[col];
            end_k = NS->idx1[col+1];
            for (int k = start_k; k < end_k; k++)
            {
                col_k = NS->idx2[k];
                val = NS->vals[k] * row_coarse[col_k];
                if (val * sign < 0)
                    coarse_sum += val;
            }
            if (fabs(coarse_sum) < zero_tol)
            {
                weak_sum += A->vals[j];
                row_strong[col] = 0;
            }
            else
            {
                row_strong[col] /= coarse_sum;
            }
        }

        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            if (row_strong[col])
            {
                start_k = SS->idx1[col];
                end_k = SS->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = SS->idx2[k];
                    val = SS->vals[k] * row_coarse[col_k];
                    if (val * sign < 0)
                    {
                        P->vals[pos[col_k]] += (row_strong[col] * val);
                    }
                }
                start_k = NS->idx1[col];
                end_k = NS->idx1[col+1];
                for (int k = start_k; k < end_k; k++)
                {
                    col_k = NS->idx2[k];
                    val = NS->vals[k] * row_coarse[col_k];
                    if (val * sign < 0)
                    {
                        P->vals[pos[col_k]] += (row_strong[col] * val);
                    }
                }
            }
        }

        for (int j = startSS; j < endSS; j++)
        {
            col = SS->idx2[j];
            idx = pos[col];
            P->vals[idx] /= -weak_sum;
        }

        for (int j = startSS; j < endSS; j++)
        {
            col = SS->idx2[j];
            pos[col] = -1;
            row_coarse[col] = 0;
        }
        for (int j = startSU; j < endSU; j++)
        {
            col = SU->idx2[j];
            row_strong[col] = 0;
        }

        P->idx1[i+1] = P->idx2.size();
    }
    P->nnz = P->idx2.size();

    delete NS;
    delete SS;
    delete SU;

    return P;
}

CSRMatrix* direct_interpolation(CSRMatrix* A,
        CSRMatrix* S, const aligned_vector<int>& states)
{
    int start, end, col;
    int ctr;
    double sum_strong_pos, sum_strong_neg;
    double sum_all_pos, sum_all_neg;
    double val, alpha, beta, diag;
    double neg_coeff, pos_coeff;

    A->sort();
    S->sort();
    A->move_diag();
    S->move_diag();

    // Copy entries of A into sparsity pattern of S
    aligned_vector<double> sa;
    if (S->nnz)
    {
        sa.resize(S->nnz);
    }
    for (int i = 0; i < A->n_rows; i++)
    {
        start = S->idx1[i];
        end = S->idx1[i+1];
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

    aligned_vector<int> col_to_new;
    if (A->n_cols)
    {
        col_to_new.resize(A->n_cols, -1);
    }

    ctr = 0;
    for (int i = 0; i < A->n_cols; i++)
    {
        if (states[i] == Selected)
        {
            col_to_new[i] = ctr++;
        }
    }

    CSRMatrix* P = new CSRMatrix(A->n_rows, ctr, A->nnz);

    P->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        if (states[i] == Selected)
        {
            P->idx2.emplace_back(col_to_new[i]);
            P->vals.emplace_back(1);
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
                if (states[col] == Selected)
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
                if (states[col] == Selected)
                {
                    val = sa[j];
                    P->idx2.emplace_back(col_to_new[col]);
                    if (val < 0)
                    {
                        P->vals.emplace_back(neg_coeff * val);
                    }
                    else
                    {
                        P->vals.emplace_back(pos_coeff * val);
                    }
                }
            }
        }
        P->idx1[i+1] = P->idx2.size();
    }
    P->nnz = P->idx2.size();

    return P;



}
