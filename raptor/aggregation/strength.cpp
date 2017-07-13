#include "core/par_matrix.hpp"

using namespace raptor;

// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)

ParCSRMatrix* ParCSRMatrix::strength(double theta)
{
    int row_start, row_end;
    int col;
    double theta_sq = theta * theta;
    double eps_diag;
    double val, val_sq;

    ParCSRMatrix* S = new ParCSRMatrix(global_num_rows, global_num_cols,
            local_num_rows, local_num_cols, first_local_row, first_local_col);
    
    if (on_proc->nnz)
    {
        S->on_proc->idx2.reserve(on_proc->nnz);
        S->on_proc->vals.reserve(on_proc->nnz);
    }
    if (off_proc->nnz)
    {
        S->off_proc->idx2.reserve(off_proc->nnz);
        S->off_proc->vals.reserve(off_proc->nnz);
    }

    std::vector<double> abs_diag;
    if (local_num_rows)
    {
        abs_diag.resize(local_num_rows);
    }
    for (int i = 0; i < local_num_rows; i++)
    {
        row_start = on_proc->idx1[i];
        row_end = on_proc->idx1[i+1];
        if (row_end - row_start)
        {
            abs_diag[i] = fabs(on_proc->vals[row_start]);
        }
        else
        {
            abs_diag[i] = 0.0;
        }
    }

    S->on_proc->idx1[0] = 0;
    S->off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        row_start = on_proc->idx1[i];
        row_end = on_proc->idx1[i+1];
        if (row_end - row_start)
        {
            eps_diag = abs_diag[i] * theta_sq;

            S->on_proc->idx2.push_back(on_proc->idx2[row_start]);
            S->on_proc->vals.push_back(1.0);
            for (int j = row_start + 1; j < row_end; j++)
            {
                col = on_proc->idx2[j];
                val = on_proc->vals[j];
                val_sq = val*val;

                if (val_sq >= eps_diag * abs_diag[col])
                {
                    S->on_proc->idx2.push_back(col);
                    S->on_proc->vals.push_back(val);
                }
            }
            
            row_start = off_proc->idx1[i];
            row_end = off_proc->idx1[i+1];
            for (int j = row_start; j < row_end; j++)
            {
                col = off_proc->idx2[j];
                val = off_proc->vals[j];
                val_sq = val*val;
                
                if (val_sq >= eps_diag * abs_diag[col])
                {
                    S->off_proc->idx2.push_back(col);
                    S->off_proc->vals.push_back(val);
                }
            }
        }
        S->on_proc->idx1[i+1] = S->on_proc->idx2.size();
        S->off_proc->idx1[i+1] = S->off_proc->idx2.size();
    }
    S->on_proc->nnz = S->on_proc->idx2.size();
    S->off_proc->nnz = S->off_proc->idx2.size();

    S->off_proc_num_cols = off_proc_num_cols;
    if (off_proc_num_cols)
    {
        S->off_proc_column_map.reserve(off_proc_num_cols);
        for (std::vector<int>::iterator it = off_proc_column_map.begin();
                it != off_proc_column_map.end(); ++it)
        {
            S->off_proc_column_map.push_back(*it);
        }
    }
    S->local_nnz = S->on_proc->nnz + S->off_proc->nnz;

    // Can copy A's comm pkg... may not need to communicate everything in comm,
    // but this is probably less costly than creating a new communicator
    if (comm)
    {
        S->comm = new ParComm((ParComm*) comm);
    }

    if (tap_comm)
    {
        S->tap_comm = new TAPComm((TAPComm*) comm);
    }

    return S;
}
