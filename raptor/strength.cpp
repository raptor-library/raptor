#include "core/par_matrix.hpp"

using namespace raptor;

// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)
ParCSRMatrix* ParCSRMatrix::strength(double theta)
{
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int col;
    double val, abs_val;
    double row_max;

    ParCSRMatrix* S = new ParCSRMatrix(partition);
    
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

    S->on_proc->idx1[0] = 0;
    S->off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        row_start_on = on_proc->idx1[i];
        row_end_on = on_proc->idx1[i+1];
        row_start_off = off_proc->idx1[i];
        row_end_off = off_proc->idx1[i+1];
        if (row_end_on - row_start_on || row_end_off - row_start_off)
        {
            // Find value with max magnitude in row
            row_max = 0.0;
            for (int j = row_start_on + 1; j < row_end_on; j++)
            {
                col = on_proc->idx2[j];
                val = on_proc->vals[j];
                abs_val = fabs(val);
                if (abs_val > row_max)
                {
                    row_max = abs_val;
                }
            } 
            for (int j = row_start_off; j < row_end_off; j++)
            {
                col = off_proc->idx2[j];
                val = off_proc->vals[j];
                abs_val = fabs(val);
                if (abs_val > row_max)
                {
                    row_max = abs_val;
                } 
            }

            // Multiply row max magnitude by theta
            row_max *= theta;

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to 
            // row_max * theta
            for (int j = row_start_on + 1; j < row_end_on; j++)
            {
                val = on_proc->vals[j];
                if (fabs(val) >= row_max)
                {
                    col = on_proc->idx2[j];
                    S->on_proc->idx2.push_back(col);
                    S->on_proc->vals.push_back(1.0);
                }
            }
            for (int j = row_start_off; j < row_end_off; j++)
            {
                val = off_proc->vals[j];
                if (fabs(val) >= row_max)
                {
                    col = off_proc->idx2[j];
                    S->off_proc->idx2.push_back(col);
                    S->off_proc->vals.push_back(1.0);
                }
            }

        }
        S->on_proc->idx1[i+1] = S->on_proc->idx2.size();
        S->off_proc->idx1[i+1] = S->off_proc->idx2.size();
    }
    S->on_proc->nnz = S->on_proc->idx2.size();
    S->off_proc->nnz = S->off_proc->idx2.size();

    S->off_proc->n_cols = off_proc_num_cols;

    std::copy(off_proc_column_map.begin(), off_proc_column_map.end(),
            std::back_inserter(S->off_proc_column_map));
    std::copy(on_proc_column_map.begin(), on_proc_column_map.end(),
            std::back_inserter(S->on_proc_column_map));
    std::copy(local_row_map.begin(), local_row_map.end(),
            std::back_inserter(S->local_row_map));

    S->off_proc_num_cols = off_proc_column_map.size();
    S->on_proc_num_cols = S->on_proc_column_map.size();
    S->local_num_rows = local_num_rows;

    S->map_partition_to_local();

    S->local_nnz = S->on_proc->nnz + S->off_proc->nnz;

    // Can copy A's comm pkg... may not need to communicate everything in comm,
    // but this is probably less costly than creating a new communicator
    if (comm)
    {
        S->comm = new ParComm((ParComm*) comm);
    }

    if (tap_comm)
    {
        S->tap_comm = new TAPComm((TAPComm*) tap_comm);
    }

    return S;

}


/*ParCSRMatrix* ParCSRMatrix::symmetric_strength(double theta)
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
                    //S->on_proc->vals.push_back(val);
                    S->on_proc->vals.push_back(1.0);
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
                    //S->off_proc->vals.push_back(val);
                    S->off_proc->vals.push_back(1.0);
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
}*/
