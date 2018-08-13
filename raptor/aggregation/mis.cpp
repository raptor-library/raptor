// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "aggregation/mis.hpp"

void mis2(CSRMatrix* A, aligned_vector<int>& states,
        double* rand_vals)
{
    int start, end, col;
    int start_k, end_k;
    int remaining, iterate;
    int ctr, v, w, u, tmp;
    bool found;

    aligned_vector<double> r;
    aligned_vector<int> V;
    aligned_vector<int> C;
    aligned_vector<int> next;
    if (A->n_rows)
    {
        r.resize(A->n_rows);
        C.resize(A->n_rows, 0);
        next.resize(A->n_rows, -1);
        V.resize(A->n_rows);
        std::iota(V.begin(), V.end(), 0);
        states.resize(A->n_rows);
        std::fill(states.begin(), states.end(), -1);
    }

    // Set random values
    if (rand_vals)
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            r[i] = rand_vals[i];
        }
    }
    else
    {
        for (int i = 0; i < A->n_rows; i++)
        {
            srand(i);
            r[i] = ((double) rand()) / RAND_MAX;
        }
    }

    // Create D (boolean matrix of directed graph)
    CSRMatrix* D = new CSRMatrix(A->n_rows, A->n_cols);
    D->idx2.reserve(0.5*A->nnz);
    D->vals.clear();
    D->idx1[0] = 0;
    for (int i = 0; i < A->n_rows; i++)
    {
        start = A->idx1[i];
        end = A->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            col = A->idx2[j];
            if (r[i] > r[col])
            {
                D->idx2.push_back(col);
            }
        }

        D->idx1[i+1] = D->idx2.size();
    }
    D->nnz = D->idx2.size();

    // Create column-wise A
    //CSCMatrix* A_csc = new CSCMatrix(A);
    CSCMatrix* A_csc = A->to_CSC();

    // Main MIS2 Loop
    remaining = A->n_rows;
    iterate = 0;
    while (remaining)
    {
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            found = false;

            start = D->idx1[v];
            end = D->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = D->idx2[j];
                if (states[w] < 0)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                states[v] = -2;
            }
        }

        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] != -2)
            {
                continue;
            }
            found = false;

            start = A->idx1[v];
            end = A->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = A->idx2[j];
                start_k = A->idx1[w];
                end_k = A->idx1[w+1];
                for (int k = start_k; k < end_k; k++)
                {
                    u = A->idx2[k];
                    if (states[u] < -1 && r[u] > r[v])
                    {
                        found = true;
                        break;
                    }
                }

                if (found)
                {
                    break;
                }
            }
            if (!found)
            {
                states[v] = -3;
            }
        }

        int head = -2;
        int length = 0;
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] == -3)
            {
                start = A_csc->idx1[v];
                end = A_csc->idx1[v+1];
                for (int j = start; j < end; j++)
                {
                    w = A_csc->idx2[j];
                    C[w] = 1;
                    if (next[w] == -1)
                    {
                        next[w] = head;
                        head = w;
                        length++;
                    }
                }
            }
        }
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] == -3)
            {
                continue;
            }
            found = false;

            start = A->idx1[v];
            end = A->idx1[v+1];
            for (int j = start; j < end; j++)
            {
                w = A->idx2[j];
                if (states[w] == -3)
                {
                    found = true;
                    break;
                }
                if (C[w])
                {
                    found = true;
                    break;
                }
            }
            if (found) 
            {
                states[v] = -4;
            }
        }
        for (int i = 0; i < length; i++)
        {
            tmp = head;
            head = next[head];
            C[tmp] = 0;
            next[tmp] = -1;
        }

        ctr = 0;
        for (int i = 0; i < remaining; i++)
        {
            v = V[i];
            if (states[v] == -3)
            {
                states[v] = 1;
            }
            else if (states[v] == -4)
            {
                states[v] = 0;
            }
            else
            {
                V[ctr++] = v;
            }
        }
        remaining = ctr;
    }
    
    delete D;
    delete A_csc;
}

