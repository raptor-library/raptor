// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_candidates.hpp"

namespace raptor {
// TODO -- currently assumes B with single candidate
ParCSRMatrix* fit_candidates(ParCSRMatrix* A, 
        const int n_aggs, const std::vector<int>& aggregates, 
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm, double tol)
{
    int rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    // Currently only implemented for this
    assert(num_candidates == 1);
    
    int col_start, col_end, row;
    int global_col, local_col;
    double val, scale;
    CommPkg* comm;

    // Calculate off_proc_column_map and num off_proc cols
    int off_proc_num_cols;
    std::set<int> off_proc_col_set;
    std::vector<int> off_proc_column_map;
    for (std::vector<int>::const_iterator it = aggregates.begin();
            it != aggregates.end(); ++it)
    {
        if (*it < 0) continue;

        if (*it < A->partition->first_local_col || *it > A->partition->last_local_col)
        {
            off_proc_col_set.insert(*it);
        }
    } 
    std::map<int, int> global_to_local;
    for (std::set<int>::iterator it = off_proc_col_set.begin();
            it != off_proc_col_set.end(); ++it)
    {
        global_to_local[*it] = off_proc_column_map.size();
        off_proc_column_map.emplace_back(*it);
    }
    off_proc_num_cols = off_proc_column_map.size();

    std::vector<int> on_proc_cols(A->on_proc_num_cols, 0);
    // Create AggOp matrices
    auto on_proc_partition_to_col = A->map_partition_to_local();
    CSRMatrix* AggOp_on = new CSRMatrix(A->local_num_rows, -1);
    CSRMatrix* AggOp_off = new CSRMatrix(A->local_num_rows, -1);
    AggOp_on->idx1[0] = 0;
    AggOp_off->idx1[0] = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        global_col = aggregates[i];
        if (global_col >= 0)
        {
            if (global_col >= A->partition->first_local_col &&
                    global_col <= A->partition->last_local_col)
            {
                local_col = on_proc_partition_to_col[global_col - A->partition->first_local_col];
                on_proc_cols[local_col] = 1;
                AggOp_on->idx2.emplace_back(local_col);
                AggOp_on->vals.emplace_back(1.0);
            }
            else
            {
                AggOp_off->idx2.emplace_back(global_to_local[global_col]);
                AggOp_off->vals.emplace_back(1.0);
            }
        }
        AggOp_on->idx1[i+1] = AggOp_on->idx2.size();
        AggOp_off->idx1[i+1] = AggOp_off->idx2.size();
    }
    AggOp_on->nnz = AggOp_on->idx2.size();
    AggOp_off->nnz = AggOp_off->idx2.size();

    // Initialize CSC Matrix for tentative interpolation
    int global_num_cols;
    RAPtor_MPI_Allreduce(&n_aggs, &global_num_cols, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    ParCSCMatrix* T_csc = new ParCSCMatrix(A->partition, A->global_num_rows, global_num_cols, 
            A->local_num_rows, n_aggs, off_proc_num_cols);
        
    T_csc->off_proc_column_map.resize(off_proc_num_cols);
    std::copy(off_proc_column_map.begin(), off_proc_column_map.end(),
            T_csc->off_proc_column_map.begin());
    T_csc->local_row_map = A->get_local_row_map();

    // Map on proc columns to new, contiguous cols
    // Create on_proc_column_map of T
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (on_proc_cols[i])
        {
            on_proc_cols[i] = T_csc->on_proc_column_map.size();
            T_csc->on_proc_column_map.emplace_back(A->on_proc_column_map[i]);
        }
    }
    for (int i = 0; i < AggOp_on->nnz; i++)
    {
        AggOp_on->idx2[i] = on_proc_cols[AggOp_on->idx2[i]];
    }
    AggOp_on->n_cols = n_aggs;
    AggOp_off->n_cols = off_proc_num_cols;

    // Convert AggOp matrices to CSC
    CSCMatrix* AggOp_on_csc = AggOp_on->to_CSC();
    CSCMatrix* AggOp_off_csc = AggOp_off->to_CSC();
    delete AggOp_on;
    delete AggOp_off;

    // Set near nullspace candidates in R to 0
    R.resize(n_aggs);
    for (int i = 0; i < n_aggs; i++)
    {
        R[i] = 0.0;
    }

    std::vector<double> off_proc_norms(off_proc_num_cols, 0);
    // Add columns of B to T (corresponding to pattern in AggOp)
    // Add on_process columns
    T_csc->on_proc->idx1[0] = 0;
    for (int i = 0; i < n_aggs; i++)
    {
        col_start = AggOp_on_csc->idx1[i];
        col_end = AggOp_on_csc->idx1[i+1];
        for (int k = col_start; k < col_end; k++)
        {
            row = AggOp_on_csc->idx2[k];
            val = B[row];
            T_csc->on_proc->idx2.emplace_back(row);
            T_csc->on_proc->vals.emplace_back(val);
            R[i] += (val * val);
        }
        T_csc->on_proc->idx1[i + 1] = T_csc->on_proc->idx2.size();
    }
    T_csc->on_proc->nnz = T_csc->on_proc->idx2.size();
    delete AggOp_on_csc;

    // Add off_process columns
    T_csc->off_proc->idx1[0] = 0;
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        col_start = AggOp_off_csc->idx1[i];
        col_end = AggOp_off_csc->idx1[i+1];
        for (int k = col_start; k < col_end; k++)
        {
            row = AggOp_off_csc->idx2[k];
            val = B[row];
            T_csc->off_proc->idx2.emplace_back(row);
            T_csc->off_proc->vals.emplace_back(val);
            off_proc_norms[i] += (val * val);
        }
        T_csc->off_proc->idx1[i + 1] = T_csc->off_proc->idx2.size();
    }
    T_csc->off_proc->nnz = T_csc->off_proc->idx2.size();
    delete AggOp_off_csc;

    // Create communicator
    if (tap_comm)
    {
        T_csc->tap_comm = new TAPComm(T_csc->partition, T_csc->off_proc_column_map,
                T_csc->on_proc_column_map, true, A->comm->mpi_comm);
        comm = T_csc->tap_comm;
    }
    else
    {
        T_csc->comm = new ParComm(T_csc->partition, T_csc->off_proc_column_map,
                T_csc->on_proc_column_map, A->comm->key, A->comm->mpi_comm);
        comm = T_csc->comm;
    }

    std::function<double(double, double)> func = &sum_func<double, double>;
    comm->communicate_T(off_proc_norms, R, 1, func, func);

    for (int i = 0; i < n_aggs; i++)
    {
        R[i] = sqrt(R[i]);
        scale = 1.0 / R[i];

        col_start = T_csc->on_proc->idx1[i];
        col_end = T_csc->on_proc->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            T_csc->on_proc->vals[j] *= scale;
        }
    }
    
    std::vector<double>& off_proc_R = comm->communicate(R);

    for (int i = 0; i < off_proc_num_cols; i++)
    {
        scale = 1.0 / off_proc_R[i];

        col_start = T_csc->off_proc->idx1[i];
        col_end = T_csc->off_proc->idx1[i+1];
        for (int j = col_start; j < col_end; j++)
        {
            T_csc->off_proc->vals[j] *= scale;
        }
    }

    ParCSRMatrix* T = T_csc->to_ParCSR();
    delete T_csc;

    return T;
}

}
