// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/par_matrix.hpp"

using namespace raptor;

// Assumes B and R are previously allocated
// B - (rows_aggop, num_candidates)
// R - (cols_aggop, num_candidates)
ParCSRMatrix* ParCSRMatrix::fit_candidates(data_t* B, data_t* R, int num_candidates, 
        double tol)
{
    CSCMatrix* on_proc_csc = new CSCMatrix(on_proc);
    CSCMatrix* off_proc_csc = new CSCMatrix(off_proc);

    ParCSCMatrix* T_csc = new ParCSCMatrix(global_num_rows, 
            global_num_cols * num_candidates, local_num_rows,
            local_num_cols * num_candidates, first_local_row,
            first_local_col * num_candidates);

    T_csc->on_proc->idx2.reserve(T_csc->on_proc->nnz * num_candidates);
    T_csc->on_proc->vals.reserve(T_csc->on_proc->nnz * num_candidates);
    T_csc->off_proc->idx2.reserve(T_csc->off_proc->nnz * num_candidates);
    T_csc->off_proc->vals.reserve(T_csc->off_proc->nnz * num_candidates);

    // Set near nullspace candidates in R to 0
    for (int i = 0; i < local_num_cols; i++)
    {
        for (int j = 0; j < num_candidates; j++)
        {
            R[i*num_candidates + j] = 0;
        }
    }

    // Add columns of B to T (corresponding to pattern in AggOp)
    T_csc->on_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_cols; i++)
    {
        col_start = on_proc_csc->idx1[i];
        col_end = on_proc_csc->idx1[i+1];
        for (int j = 0; j < num_candidates; j++)
        {
            for (int k = col_start; k < col_end; k++)
            {
                row = on_proc_csc->idx2[j];
                idx_B = (j * local_num_rows) + row;
                val = B[idx_B];
                if (fabs(val) > zero_tol)
                {
                    T_csc->on_proc->idx2.push_back(row);
                    T_csc->on_proc->vals.push_back(val);
                }
            }
            T_csc->on_proc->idx1[(i*num_candidates)+j+1] = T_csc->on_proc->idx2.size();
        }
    }
    T_csc->on_proc->nnz = T_csc->on_proc->idx2.size();

    T_csc->on_proc->idx1[0] = 0;
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        col_start = off_proc_csc->idx1[i];
        col_end = off_proc_csc->idx1[i+1];
        for (int j = 0; j < num_candidates; j++)
        {
            for (int k = col_start; k < col_end; k++)
            {
                row = off_proc_csc->idx2[j];
                idx_B = (j * local_num_rows) + row;
                val = B[idx_B];
                if (fabs(val) > zero_tol)
                {
                    T_csc->off_proc->idx2.push_back(row);
                    T_csc->off_proc->vals.push_back(val);
                }
            }
            T_csc->off_proc->idx1[(i*num_candidates)+j+1] = T_csc->off_proc->idx2.size();
        }
    }
    T_csc->off_proc->nnz = T_csc->off_proc->idx2.size();
      
    for (int i = 0; i < local_num_cols; i++)
    {
        int idx_R = i * num_candidates * num_candidates;

        // Need to calculate the norm of all columns

        for (int j = 0; j < num_candidates; j++)
        {
            double norm_j = 0;
            int col_start_j = T_csc->idx1[i*num_candidates + j];
            int col_end_j = T_csc->idx1[i*num_candidates + j + 1];
            int col_size = col_end_j - col_start_j;

            // Calculate norm of column
            for (int k = col_start_j; k < col_end_j; k++)
            {
                double val = T_csc->vals[k];
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
                double val = T_csc->vals[k];
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

    CSRMatrix* T = new CSRMatrix(T_csc);
    delete T_csc;
    return T;
}

