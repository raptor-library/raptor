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
    double row_scale;
    double threshold;
    double diag;

    ParCSRMatrix* S = new ParCSRMatrix(partition, global_num_rows, global_num_cols,
            local_num_rows, on_proc_num_cols, off_proc_num_cols);

    std::vector<bool> col_exists;
    if (off_proc_num_cols)
    {
        col_exists.resize(off_proc_num_cols, false);
    }

    sort();
    on_proc->move_diag();
    
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
            if (on_proc->idx2[row_start_on] == i)
            {
                diag = on_proc->vals[row_start_on];
                row_start_on++;
            }
            else
            {
                diag = 0.0;
            }

            // Find value with max magnitude in row
            if (diag < 0.0)
            {
                row_scale = -RAND_MAX; 
                for (int j = row_start_on; j < row_end_on; j++)
                {
                    val = on_proc->vals[j];
                    if (val > row_scale)
                    {
                        row_scale = val;
                    }
                }    
                for (int j = row_start_off; j < row_end_off; j++)
                {
                    val = off_proc->vals[j];
                    if (val > row_scale)
                    {
                        row_scale = val;
                    }
                } 
            }
            else
            {
                row_scale = RAND_MAX;
                for (int j = row_start_on; j < row_end_on; j++)
                {
                    val = on_proc->vals[j];
                    if (val < row_scale)
                    {
                        row_scale = val;
                    }
                }    
                for (int j = row_start_off; j < row_end_off; j++)
                {
                    val = off_proc->vals[j];
                    if (val < row_scale)
                    {
                        row_scale = val;
                    }
                } 
            }

            // Multiply row max magnitude by theta
            threshold = row_scale * theta;

            // Always add diagonal
            S->on_proc->idx2.push_back(i);
            S->on_proc->vals.push_back(diag);

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to 
            // row_max * theta
            if (diag < 0)
            {
                for (int j = row_start_on; j < row_end_on; j++)
                {
                    val = on_proc->vals[j];
                    if (val > threshold)
                    {
                        S->on_proc->idx2.push_back(on_proc->idx2[j]);
                        S->on_proc->vals.push_back(on_proc->vals[j]);
                    }
                }
                for (int j = row_start_off; j < row_end_off; j++)
                {
                    val = off_proc->vals[j];
                    if (val > threshold)
                    {
                        col = off_proc->idx2[j];
                        S->off_proc->idx2.push_back(col);
                        S->off_proc->vals.push_back(off_proc->vals[j]);
                        col_exists[col] = true;
                    }
                }
            }
            else
            {
                for (int j = row_start_on; j < row_end_on; j++)
                {
                    val = on_proc->vals[j];
                    if (val < threshold)
                    {
                        S->on_proc->idx2.push_back(on_proc->idx2[j]);
                        S->on_proc->vals.push_back(on_proc->vals[j]);
                    }
                }
                for (int j = row_start_off; j < row_end_off; j++)
                {
                    val = off_proc->vals[j];
                    if (val < threshold)
                    {
                        col = off_proc->idx2[j];
                        S->off_proc->idx2.push_back(col);
                        S->off_proc->vals.push_back(off_proc->vals[j]);
                        col_exists[col] = true;
                    }
                }
            }

        }
        S->on_proc->idx1[i+1] = S->on_proc->idx2.size();
        S->off_proc->idx1[i+1] = S->off_proc->idx2.size();
    }
    S->on_proc->nnz = S->on_proc->idx2.size();
    S->off_proc->nnz = S->off_proc->idx2.size();
    S->local_nnz = S->on_proc->nnz + S->off_proc->nnz;

    S->on_proc_column_map = get_on_proc_column_map();
    S->local_row_map = get_local_row_map();

    std::vector<int> orig_to_S;
    if (off_proc_num_cols)
    {
        orig_to_S.resize(off_proc_num_cols, -1);
    }
    S->off_proc_column_map.reserve(off_proc_num_cols);
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            orig_to_S[i] = S->off_proc_column_map.size();
            S->off_proc_column_map.push_back(off_proc_column_map[i]);
        }
    }
    S->off_proc_num_cols = S->off_proc_column_map.size();
    for (std::vector<int>::iterator it = S->off_proc->idx2.begin();
            it != S->off_proc->idx2.end(); ++it)
    {
        *it = orig_to_S[*it];
    }

    // Can copy A's comm pkg... may not need to communicate everything in comm,
    // but this is probably less costly than creating a new communicator
    // TODO... but is it?
    if (comm)
    {
        S->comm = new ParComm((ParComm*) comm, orig_to_S);
    }

    if (tap_comm)
    {
        //S->tap_comm = new TAPComm((TAPComm*) tap_comm, orig_to_S);
    }

    return S;

}


