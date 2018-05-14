// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* classical_strength(ParCSRMatrix* A, double theta, int num_variables,
        int* variables)
{
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int col;
    double val;
    double row_scale;
    double threshold;
    double diag;

    ParCSRMatrix* S = new ParCSRMatrix(A->partition, A->global_num_rows, A->global_num_cols,
            A->local_num_rows, A->on_proc_num_cols, A->off_proc_num_cols);
    
    int* off_variables;
    if (num_variables > 1)
    {
        A->comm->communicate(variables);
        off_variables = A->comm->recv_data->int_buffer.data();
    }

    aligned_vector<bool> col_exists;
    if (A->off_proc_num_cols)
    {
        col_exists.resize(A->off_proc_num_cols, false);
    }

    A->sort();
    A->on_proc->move_diag();
    S->on_proc->vals.clear();
    S->off_proc->vals.clear();
    
    if (A->on_proc->nnz)
    {
        S->on_proc->idx2.reserve(A->on_proc->nnz);
    }
    if (A->off_proc->nnz)
    {
        S->off_proc->idx2.reserve(A->off_proc->nnz);
    }

    S->on_proc->idx1[0] = 0;
    S->off_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        row_start_on = A->on_proc->idx1[i];
        row_end_on = A->on_proc->idx1[i+1];
        row_start_off = A->off_proc->idx1[i];
        row_end_off = A->off_proc->idx1[i+1];
        if (row_end_on - row_start_on || row_end_off - row_start_off)
        {
            if (A->on_proc->idx2[row_start_on] == i)
            {
                diag = A->on_proc->vals[row_start_on];
                row_start_on++;
            }
            else
            {
                diag = 0.0;
            }

            // Find value with max magnitude in row
            if (num_variables == 1)
            {
                if (diag < 0.0)
                {
                    row_scale = -RAND_MAX; 
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        val = A->on_proc->vals[j];
                        if (val > row_scale)
                        {
                            row_scale = val;
                        }
                    }    
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        val = A->off_proc->vals[j];
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
                        val = A->on_proc->vals[j];
                        if (val < row_scale)
                        {
                            row_scale = val;
                        }
                    }    
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        val = A->off_proc->vals[j];
                        if (val < row_scale)
                        {
                            row_scale = val;
                        }
                    } 
                }
            }
            else
            {
                if (diag < 0.0)
                {
                    row_scale = -RAND_MAX; 
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        col = A->on_proc->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->on_proc->vals[j];
                            if (val > row_scale)
                            {
                                row_scale = val;
                            }
                        }
                    }    
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        col = A->off_proc->idx2[j];
                        if (variables[i] == off_variables[col])
                        {
                            val = A->off_proc->vals[j];
                            if (val > row_scale)
                            {
                                row_scale = val;
                            }
                        }
                    } 
                }
                else
                {
                    row_scale = RAND_MAX;
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        col = A->on_proc->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->on_proc->vals[j];
                            if (val < row_scale)
                            {
                                row_scale = val;
                            }
                        }
                    }    
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        col = A->off_proc->idx2[j];
                        if (variables[i] == off_variables[col])
                        {
                            val = A->off_proc->vals[j];
                            if (val < row_scale)
                            {
                                row_scale = val;
                            }
                        }
                    } 
                }
            }



            // Multiply row max magnitude by theta
            threshold = row_scale * theta;

            // Always add diagonal
            S->on_proc->idx2.push_back(i);

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to 
            // row_max * theta
            if (num_variables == 1)
            {
                if (diag < 0)
                {
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        val = A->on_proc->vals[j];
                        if (val > threshold)
                        {
                            S->on_proc->idx2.push_back(A->on_proc->idx2[j]);
                        }
                    }
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        val = A->off_proc->vals[j];
                        if (val > threshold)
                        {
                            col = A->off_proc->idx2[j];
                            S->off_proc->idx2.push_back(col);
                            col_exists[col] = true;
                        }
                    }
                }
                else
                {
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        val = A->on_proc->vals[j];
                        if (val < threshold)
                        {
                            S->on_proc->idx2.push_back(A->on_proc->idx2[j]);
                        }
                    }
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        val = A->off_proc->vals[j];
                        if (val < threshold)
                        {
                            col = A->off_proc->idx2[j];
                            S->off_proc->idx2.push_back(col);
                            col_exists[col] = true;
                        }
                    }
                }
            }
            else
            {
                if (diag < 0)
                {
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        col = A->on_proc->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->on_proc->vals[j];
                            if (val > threshold)
                            {
                                S->on_proc->idx2.push_back(col);
                            }
                        }
                    }
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        col = A->off_proc->idx2[j];
                        if (variables[i] == off_variables[col])
                        {
                            val = A->off_proc->vals[j];
                            if (val > threshold)
                            {
                                S->off_proc->idx2.push_back(col);
                                col_exists[col] = true;
                            }
                        }
                    }
                }
                else
                {
                    for (int j = row_start_on; j < row_end_on; j++)
                    {
                        col = A->on_proc->idx2[j];
                        if (variables[i] == variables[col])
                        {
                            val = A->on_proc->vals[j];
                            if (val < threshold)
                            {
                                S->on_proc->idx2.push_back(col);
                            }
                        }
                    }
                    for (int j = row_start_off; j < row_end_off; j++)
                    {
                        col = A->off_proc->idx2[j];
                        if (variables[i] == off_variables[col])
                        {
                            val = A->off_proc->vals[j];
                            if (val < threshold)
                            {
                                S->off_proc->idx2.push_back(col);
                                col_exists[col] = true;
                            }
                        }
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

    S->on_proc_column_map = A->get_on_proc_column_map();
    S->local_row_map = A->get_local_row_map();

    aligned_vector<int> orig_to_S;
    if (A->off_proc_num_cols)
    {
        orig_to_S.resize(A->off_proc_num_cols, -1);
    }
    S->off_proc_column_map.reserve(A->off_proc_num_cols);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            orig_to_S[i] = S->off_proc_column_map.size();
            S->off_proc_column_map.push_back(A->off_proc_column_map[i]);
        }
    }
    S->off_proc_num_cols = S->off_proc_column_map.size();
    for (aligned_vector<int>::iterator it = S->off_proc->idx2.begin();
            it != S->off_proc->idx2.end(); ++it)
    {
        *it = orig_to_S[*it];
    }

    // Can copy A's comm pkg... may not need to communicate everything in comm,
    // but this is probably less costly than creating a new communicator
    // TODO... but is it?
    if (A->comm)
    {
        S->comm = new ParComm((ParComm*) A->comm, orig_to_S);
    }

    if (A->tap_comm)
    {
        //S->tap_comm = new TAPComm((TAPComm*) tap_comm, orig_to_S);
    }

    return S;

}

ParCSRMatrix* symmetric_strength(ParCSRMatrix* A, double theta)
{
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int col;
    double val;
    double row_scale;
    double eps;
    double diag;

    aligned_vector<double> diags(A->local_num_rows);

    ParCSRMatrix* S = new ParCSRMatrix(A->partition, A->global_num_rows, A->global_num_cols,
            A->local_num_rows, A->on_proc_num_cols, A->off_proc_num_cols);
    
    aligned_vector<bool> col_exists;
    if (A->off_proc_num_cols)
    {
        col_exists.resize(A->off_proc_num_cols, false);
    }

    A->sort();
    A->on_proc->move_diag();
    S->on_proc->vals.clear();
    S->off_proc->vals.clear();
    
    if (A->on_proc->nnz)
    {
        S->on_proc->idx2.reserve(A->on_proc->nnz);
    }
    if (A->off_proc->nnz)
    {
        S->off_proc->idx2.reserve(A->off_proc->nnz);
    }

    for (int i = 0; i < A->local_num_rows; i++)
    {
        row_start_on = A->on_proc->idx1[i];
        row_end_on = A->on_proc->idx1[i+1];

        if (row_end_on > row_start_on && A->on_proc->idx2[row_start_on] == i)
        {
            diags[i] = fabs(A->on_proc->vals[row_start_on]);
        }
    }
    aligned_vector<double>& off_proc_diags = A->comm->communicate(diags);

    S->on_proc->idx1[0] = 0;
    S->off_proc->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        eps = theta * theta * diags[i];

        row_start_on = A->on_proc->idx1[i];
        row_end_on = A->on_proc->idx1[i+1];
        row_start_off = A->off_proc->idx1[i];
        row_end_off = A->off_proc->idx1[i+1];
        
        // Always add the diagonal
        if (row_end_on > row_start_on && A->on_proc->idx2[row_start_on] == i)
        {
            S->on_proc->vals.push_back(A->on_proc->vals[row_start_on]);
            row_start_on++;
        }
        else
        {
            S->on_proc->vals.push_back(0.0);
        }
        S->on_proc->idx2.push_back(i);

        // On Process Block
        for (int j = row_start_on; j < row_end_on; j++)
        {
            col = A->on_proc->idx2[j];
            val = A->on_proc->vals[j];
            if (val * val >= eps * diags[col])
            {
                S->on_proc->idx2.push_back(col);
                S->on_proc->vals.push_back(val);
            }
        }
        S->on_proc->idx1[i+1] = S->on_proc->idx2.size();
        
        // Off Process Block
        for (int j = row_start_off; j < row_end_off; j++)
        {
            col = A->off_proc->idx2[j];
            val = A->off_proc->vals[j];
            if (val * val >= eps * off_proc_diags[col])
            {
                S->off_proc->idx2.push_back(col);
                S->off_proc->vals.push_back(val);
                col_exists[col] = true;
            }
        }
        S->off_proc->idx1[i+1] = S->off_proc->idx2.size();
    }
    S->on_proc->nnz = S->on_proc->idx2.size();
    S->off_proc->nnz = S->off_proc->idx2.size();
    S->local_nnz = S->on_proc->nnz + S->off_proc->nnz;

    S->on_proc_column_map = A->get_on_proc_column_map();
    S->local_row_map = A->get_local_row_map();

    aligned_vector<int> orig_to_S;
    if (A->off_proc_num_cols)
    {
        orig_to_S.resize(A->off_proc_num_cols, -1);
    }
    S->off_proc_column_map.reserve(A->off_proc_num_cols);
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        if (col_exists[i])
        {
            orig_to_S[i] = S->off_proc_column_map.size();
            S->off_proc_column_map.push_back(A->off_proc_column_map[i]);
        }
    }
    S->off_proc_num_cols = S->off_proc_column_map.size();
    for (aligned_vector<int>::iterator it = S->off_proc->idx2.begin();
            it != S->off_proc->idx2.end(); ++it)
    {
        *it = orig_to_S[*it];
    }

    // Can copy A's comm pkg... may not need to communicate everything in comm,
    // but this is probably less costly than creating a new communicator
    // TODO... but is it?
    if (A->comm)
    {
        S->comm = new ParComm((ParComm*) A->comm, orig_to_S);
    }

    if (A->tap_comm)
    {
        //S->tap_comm = new TAPComm((TAPComm*) tap_comm, orig_to_S);
    }

    return S;
}


// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)
ParCSRMatrix* ParCSRMatrix::strength(strength_t strength_type,
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


