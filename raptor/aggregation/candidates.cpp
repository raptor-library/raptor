// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "candidates.hpp"

using namespace raptor;

// Assumes B and R are previously allocated
// B - (rows_aggop, num_candidates)
// R - (cols_aggop, num_candidates)
void fit_candidates(CSCMatrix& AggOp, CSCMatrix* T, data_t* B, data_t* R, int num_candidates, double tol)
{
    // Set near nullspace candidates in R to 0
    for (int i = 0; i < AggOp.n_cols; i++)
    {
        for (int j = 0; j < num_candidates; j++)
        {
            R[i*num_candidates + j] = 0;
        }
    }

    // Create T, copying columns of B 
    T->n_rows = AggOp.n_rows;
    T->n_cols = AggOp.n_cols * num_candidates;
    T->nnz = 0;
    T->idx1.resize(T->n_cols+1);
    T->idx2.clear();
    T->vals.clear();
    T->idx2.reserve(AggOp.nnz * num_candidates);
    T->vals.reserve(AggOp.nnz * num_candidates);

    // Add columns of B to T (corresponding to pattern in AggOp)
    T->idx1[0] = 0;
    for (int i = 0; i < AggOp.n_cols; i++)
    {
        int col_start = AggOp.idx1[i];
        int col_end = AggOp.idx1[i+1];
        for (int j = 0; j < num_candidates; j++)
        {
            for (int k = col_start; k < col_end; k++)
            {
                int row = AggOp.idx2[k];
                int idx_B = j * AggOp.n_rows + row;
                double val = B[idx_B];
                T->idx2.push_back(row);
                T->vals.push_back(val);
            }
            T->idx1[i*num_candidates + j] = T->idx2.size();
        }
    }
    T->nnz = T->idx2.size();

/*    for (int i = 0; i < AggOp.n_cols; i++)
    {
        int idx_R = i * num_candidates * num_candidates;

        for (int j = 0; j < num_candidates; j++)
        {
            double norm_j = 0;
            int col_start_j = T->idx1[i*num_candidates + j];
            int col_end_j = T->idx1[i*num_candidates + j + 1];
            int col_size = col_end_j - col_start_j;

            // Calculate norm of column
            for (int k = col_start_j; k < col_end_j; k++)
            {
                double val = T->vals[k];
                norm_j += val * val;
            }
            norm_j = sqrt(norm_j);

            // Calculate threshold (above which values are retained)
            double threshold = norm_j * tol;

            // Orthogonalize against previous candidate vectors
            for (int k = 0; k < j; k++)
            {

                // Calculate dot product with previous candidate vector            
                double dot_prod = 0;
                int col_start_k = T->idx1[i*num_candidates + k];
                for (int l = 0; l < col_size; l++)
                {
                    double val_k = T->vals[col_start_k + l];
                    double val_j = T->vals[col_start_j + l];
                    dot_prod += val_k * val_j;
                }

                // Orthogonalize against col k
                for (int l = 0; l < col_size; l++)
                {
                    double val_k = T->vals[col_start_k + l];
                    T->vals[col_start_j + l] -= dot_prod * val_k;
                }

                // Update value in R
                R[idx_R + k*num_candidates + j] = dot_prod;
            }

            // Calculate norm of column (to compare against prev threshold)
            norm_j = 0;
            for (int k = col_start_j; k < col_end_j; k++)
            {
                double val = T->vals[k];
                norm_j += val * val;
            }
            norm_j = sqrt(norm_j);

            double scale;
            if (norm_j > threshold)
            {
                scale = 1.0 / norm_j;
                R[idx_R + j*num_candidates + j] = norm_j;            
            }
            else
            {
                scale = 0;
                R[idx_R + j*num_candidates + j] = 0;            
            }

            for (int k = col_start_j; k < col_end_j; k++)
            {
                T->vals[k] *= scale;
            }
        }
    }*/
}



