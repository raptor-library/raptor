// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "candidates.hpp"

CSRMatrix* fit_candidates(const int n_aggs, const std::vector<int>& aggregates, 
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, double tol)
{
    int col_start, col_end;
    int row, idx_B;
    double val;

    // Create CSC Matrix Holding Aggregates
    int n_rows = aggregates.size();
    CSRMatrix* AggOp = new CSRMatrix(n_rows, n_aggs, n_rows);
    AggOp->idx1[0] = 0;
    for (int i = 0; i < n_rows; i++)
    {
        AggOp->idx2.emplace_back(aggregates[i]);
        AggOp->vals.emplace_back(1.0);
        AggOp->idx1[i+1] = i+1;
    }
    AggOp->nnz = n_rows;
    CSCMatrix* AggOp_csc = AggOp->to_CSC();
    delete AggOp;

    // Initialize CSC matrix for tentative interpolation
    CSCMatrix* T_csc = new CSCMatrix(n_rows, n_aggs * num_candidates, n_rows * num_candidates);
    
    // Set near nullspace candidates in R to 0
    R.resize(n_aggs);
    for (int i = 0; i < n_aggs; i++)
    {
        R[i] = 0.0;
    }

    // Add columns of B to T (corresponding to pattern in AggOp)
    T_csc->idx1[0] = 0;
    for (int i = 0; i < n_aggs; i++)
    {
        col_start = AggOp_csc->idx1[i];
        col_end = AggOp_csc->idx1[i+1];
        for (int j = 0; j < num_candidates; j++)
        {
            for (int k = col_start; k < col_end; k++)
            {
                row = AggOp_csc->idx2[k];
                idx_B = (j*n_rows) + row;
                val = B[idx_B];
                T_csc->idx2.emplace_back(row);
                T_csc->vals.emplace_back(val);
            }
            T_csc->idx1[i*num_candidates + j + 1] = T_csc->idx2.size();
        }
    }
    T_csc->nnz = T_csc->idx2.size();
    delete AggOp_csc;

    for (int i = 0; i < n_aggs; i++)
    {
        int idx_R = i * num_candidates * num_candidates;

        for (int j = 0; j < num_candidates; j++)
        {
            double norm_j = 0;
            int col_start_j = T_csc->idx1[i*num_candidates + j];
            int col_end_j = T_csc->idx1[i*num_candidates + j + 1];
            int col_size = col_end_j - col_start_j;

            // Calculate norm of column
            for (int k = col_start_j; k < col_end_j; k++)
            {
                val = T_csc->vals[k];
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
                int col_start_k = T_csc->idx1[i*num_candidates + k];
                for (int l = 0; l < col_size; l++)
                {
                    double val_k = T_csc->vals[col_start_k + l];
                    double val_j = T_csc->vals[col_start_j + l];
                    dot_prod += val_k * val_j;
                }

                // Orthogonalize against col k
                for (int l = 0; l < col_size; l++)
                {
                    double val_k = T_csc->vals[col_start_k + l];
                    T_csc->vals[col_start_j + l] -= dot_prod * val_k;
                }

                // Update value in R
                R[idx_R + k*num_candidates + j] = dot_prod;
            }

            // Calculate norm of column (to compare against prev threshold)
            norm_j = 0;
            for (int k = col_start_j; k < col_end_j; k++)
            {
                val = T_csc->vals[k];
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
                T_csc->vals[k] *= scale;
            }
        }
    }

    CSRMatrix* T = T_csc->to_CSR();
    delete T_csc;

    return T;
}
