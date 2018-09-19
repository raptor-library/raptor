// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* classical_strength(ParCSRMatrix* A, double theta, bool tap_amg, int num_variables,
        int* variables, data_t* comm_t)
{
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int col;
    double val;
    double row_scale;
    double threshold;
    double diag;

    CommPkg* comm = A->comm;
    if (tap_amg)
    {
        comm = A->tap_comm;
    }

    ParCSRMatrix* S = new ParCSRMatrix(A->partition, A->global_num_rows, A->global_num_cols,
            A->local_num_rows, A->on_proc_num_cols, A->off_proc_num_cols);
    
    int* off_variables;
    if (num_variables > 1)
    {
        if (comm_t) *comm_t -= MPI_Wtime();
        aligned_vector<int>& recvbuf = comm->communicate(variables);
        if (comm_t) *comm_t += MPI_Wtime();

        off_variables = recvbuf.data();
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
    if (!comm_t)
    {
        if (A->comm)
        {
            S->comm = new ParComm((ParComm*) A->comm, orig_to_S, comm_t);
        }

        if (A->tap_comm)
        {
            S->tap_comm = new TAPComm((TAPComm*) A->tap_comm, orig_to_S, comm_t);
        }
    }

    return S;

}

// TODO -- currently this assumes all diags are same sign...
ParCSRMatrix* symmetric_strength(ParCSRMatrix* A, double theta, bool tap_amg, data_t* comm_t)
{
    int row_start_on, row_end_on;
    int row_start_off, row_end_off;
    int col;
    double val;
    double row_scale;
    double threshold;
    double diag;

    CommPkg* comm = A->comm;
    if (tap_amg)
    {
        comm = A->tap_comm;
    }

    aligned_vector<int> neg_diags;
    aligned_vector<double> row_scales;
    if (A->local_num_rows) 
    {
        row_scales.resize(A->local_num_rows, 0);
        neg_diags.resize(A->local_num_rows);
    }

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
            if (diag < 0.0)
            {
                neg_diags[i] = 1;
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
                neg_diags[i] = 0;
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

            // Multiply row max magnitude by theta
            row_scales[i] = row_scale * theta;
        }
    }

    if (comm_t) *comm_t -= MPI_Wtime();
    aligned_vector<double>& off_proc_row_scales = comm->communicate(row_scales);
    aligned_vector<int>& off_proc_neg_diags = comm->communicate(neg_diags);
    if (comm_t) *comm_t += MPI_Wtime();
    
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
            bool neg_diag = neg_diags[i];           
            threshold = row_scales[i];

            // Always add diagonal
            S->on_proc->idx2.push_back(i);

            // Add all off-diagonal entries to strength
            // if magnitude greater than equal to 
            // row_max * theta
            for (int j = row_start_on; j < row_end_on; j++)
            {
                val = A->on_proc->vals[j];
                col = A->on_proc->idx2[j];
                if ((neg_diag && val > threshold) || (!neg_diag && val < threshold) 
                        || (neg_diags[col] && val > row_scales[col])
                        || (!neg_diags[col] && val < row_scales[col]))
                {
                    S->on_proc->idx2.push_back(col);
                }
            }
            for (int j = row_start_off; j < row_end_off; j++)
            {
                val = A->off_proc->vals[j];
                col = A->off_proc->idx2[j];
                if ((neg_diag && val > threshold) || (!neg_diag && val < threshold)
                        || (off_proc_neg_diags[col] && val > off_proc_row_scales[col])
                        || (!off_proc_neg_diags[col] && val < off_proc_row_scales[col]))
                {
                    S->off_proc->idx2.push_back(col);
                    col_exists[col] = true;
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
    if (!comm_t)
    {
    	if (A->comm)
    	{
            S->comm = new ParComm((ParComm*) A->comm, orig_to_S, comm_t);
	}

    	if (A->tap_comm)
    	{
            S->tap_comm = new TAPComm((TAPComm*) A->tap_comm, orig_to_S, comm_t);
   	}
    }

    return S;
}


// Assumes ParCSRMatrix is previously sorted
// TODO -- have ParCSRMatrix bool sorted (and sort if not previously)
ParCSRMatrix* ParCSRMatrix::strength(strength_t strength_type,
        double theta, bool tap_amg, int num_variables, int* variables,
        data_t* comm_t)
{
    switch (strength_type)
    {
        case Classical:
            return classical_strength(this, theta, tap_amg, num_variables, variables, comm_t);
        case Symmetric:
            return symmetric_strength(this, theta, tap_amg, comm_t);
    }
}

