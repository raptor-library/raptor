// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_candidates.hpp"

namespace raptor {

static ParCSRMatrix* fit_candidates_csr(ParCSRMatrix* A,
        const int n_aggs, const std::vector<int>& aggregates, 
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm, double tol)
{
    assert(B.size() == A->local_num_rows*num_candidates);
    
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

    std::vector<int> on_proc_agg_marker(A->on_proc_num_cols, 0);
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
                on_proc_agg_marker[local_col] = 1;
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
    int global_num_aggs;
    RAPtor_MPI_Allreduce(&n_aggs, &global_num_aggs, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    ParCSCMatrix* T_csc = new ParCSCMatrix(A->partition, A->global_num_rows, global_num_aggs*num_candidates, A->local_num_rows, n_aggs*num_candidates, off_proc_num_cols*num_candidates);
        
    T_csc->off_proc_column_map.resize(off_proc_num_cols*num_candidates);

    int count = 0;
    for (auto col: off_proc_column_map)
    {
        for (int i = 0; i < num_candidates; i++)
        {
            T_csc->off_proc_column_map[count++] = col*num_candidates+i;
        }
    }

    T_csc->local_row_map = A->get_local_row_map();

    // Map on-proc aggregate roots to compact local aggregate indices
    // candidate-expanded T columns are created later
    std::vector<int> on_proc_agg_to_local_agg(A->on_proc_num_cols, -1);
    std::vector<int> on_proc_aggs;
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (on_proc_agg_marker[i])
        {
            on_proc_agg_to_local_agg[i] = on_proc_aggs.size();
            on_proc_aggs.emplace_back(A->on_proc_column_map[i]);
        }
    }
    for (int i = 0; i < AggOp_on->nnz; i++)
    {
        AggOp_on->idx2[i] = on_proc_agg_to_local_agg[AggOp_on->idx2[i]];
    }
    AggOp_on->n_cols = n_aggs;
    AggOp_off->n_cols = off_proc_num_cols;

    // expand on-proc columns with candidates for T_csc
    T_csc->on_proc_column_map.reserve(n_aggs*num_candidates);
    for (int i = 0; i < n_aggs; i++)
    {
        int col = on_proc_aggs[i];

        for (int j = 0; j < num_candidates; j++)
        {
            T_csc->on_proc_column_map.emplace_back(col*num_candidates+j);
        }
    }

    // Convert AggOp matrices to CSC
    CSCMatrix* AggOp_on_csc = AggOp_on->to_CSC();
    CSCMatrix* AggOp_off_csc = AggOp_off->to_CSC();
    delete AggOp_on;
    delete AggOp_off;

    // Add columns of B to T (corresponding to pattern in AggOp)
    // Add on_process columns
    T_csc->on_proc->idx1[0] = 0;
    for (int i = 0; i < n_aggs; i++)
    {
        col_start = AggOp_on_csc->idx1[i];
        col_end = AggOp_on_csc->idx1[i+1];
        for (int j = 0; j < num_candidates; j++)
        {
            auto T_col = i*num_candidates+j;
            for (int k = col_start; k < col_end; k++)
            {
                row = AggOp_on_csc -> idx2[k];
                // B is stored column-major
                val = B[j*A->local_num_rows+row];
                T_csc->on_proc->idx2.emplace_back(row);
                T_csc->on_proc->vals.emplace_back(val);
            }
            T_csc->on_proc->idx1[T_col + 1] = T_csc->on_proc->idx2.size();
        }
       
    }
    T_csc->on_proc->nnz = T_csc->on_proc->idx2.size();
    delete AggOp_on_csc;

    // Add off_process columns
    T_csc->off_proc->idx1[0] = 0;
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        col_start = AggOp_off_csc->idx1[i];
        col_end = AggOp_off_csc->idx1[i+1];
        for (int j = 0; j < num_candidates; j++)
        {
            auto T_col = i*num_candidates+j;
            for (int k = col_start; k < col_end; k++)
            {
                row = AggOp_off_csc->idx2[k];
                val = B[j*A->local_num_rows+row];
                T_csc->off_proc->idx2.emplace_back(row);
                T_csc->off_proc->vals.emplace_back(val);
            }
            T_csc->off_proc->idx1[T_col + 1] = T_csc->off_proc->idx2.size();
        }
    }
    T_csc->off_proc->nnz = T_csc->off_proc->idx2.size();
    delete AggOp_off_csc;

    // QR
    R.resize(n_aggs*num_candidates*num_candidates, 0.0);
    std::vector<double> on_proc_norms(n_aggs*num_candidates, 0);
    std::vector<double> off_proc_norms(off_proc_num_cols*num_candidates, 0);
    std::vector<double> threshold(n_aggs, 0); 
    std::function<double(double, double)> func = &sum_func<double, double>;

    // helper to convert flat index for (R_agg)[i,j]
    auto R_flat_idx = [&](int agg_idx, int i, int j) 
    {
        return agg_idx*num_candidates*num_candidates+ i*num_candidates+j;
    };

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

    for (int i = 0; i < num_candidates; i++)
    {
        std::fill(on_proc_norms.begin(), on_proc_norms.end(), 0.0);
        std::fill(off_proc_norms.begin(), off_proc_norms.end(), 0.0);

        // compute original norm of candidate i on_proc 
        for (int j = 0; j < n_aggs; j++)
        {
            int T_col = j*num_candidates+i;
            col_start = T_csc -> on_proc->idx1[T_col];
            col_end = T_csc -> on_proc->idx1[T_col+1];
            // add up the norms
            double norm2 = 0.0;
            for (int k = col_start; k< col_end; k++)
            {
                auto val = T_csc -> on_proc->vals[k];
                norm2 += val*val;
            }
            on_proc_norms[T_col] = norm2;
        }

        // compute original norm of candidate i off_proc 
        for (int j = 0; j < off_proc_num_cols; j++)
        {
            int T_col = j*num_candidates+i;
            col_start = T_csc -> off_proc->idx1[T_col];
            col_end = T_csc -> off_proc->idx1[T_col+1];
            // add up the norms 
            double norm2 = 0.0;
            for (int k = col_start; k< col_end; k++)
            {
                auto val = T_csc -> off_proc->vals[k];
                norm2 += val*val;
            }
            off_proc_norms[T_col] = norm2;
        }

        comm->communicate_T(off_proc_norms, on_proc_norms, 1, func, func);

        for (int j = 0; j < n_aggs; j++)
        {
            threshold[j] = tol*sqrt(on_proc_norms[j*num_candidates+i]);
        }

        // orthogonalize against previous candidates
        for (int j = 0; j < i; j++)
        {
            std::fill(on_proc_norms.begin(), on_proc_norms.end(), 0.0);
            std::fill(off_proc_norms.begin(), off_proc_norms.end(), 0.0);
            // on_proc
            for (int k = 0; k< n_aggs; k++)
            {
                int prev_col = k*num_candidates+j;
                int curr_col = k*num_candidates+i;

                int prev_col_start = T_csc -> on_proc->idx1[prev_col];
                int prev_col_end = T_csc -> on_proc->idx1[prev_col +1];
                int curr_col_start = T_csc -> on_proc->idx1[curr_col];

                double dot_prod = 0.0;
                for (int l = prev_col_start; l<prev_col_end; l++)
                {
                    // here assume each aggregate block is dense hence 0 is stored explicitly
                    dot_prod += (T_csc -> on_proc->vals[l]) * (T_csc -> on_proc->vals[l - prev_col_start + curr_col_start]);
                }

                on_proc_norms[curr_col] = dot_prod;
            }

            // off_proc
            for (int k = 0; k< off_proc_num_cols; k++)
            {
                int prev_col = k*num_candidates+j;
                int curr_col = k*num_candidates+i;

                int prev_col_start = T_csc -> off_proc->idx1[prev_col];
                int prev_col_end = T_csc -> off_proc->idx1[prev_col +1];
                int curr_col_start = T_csc -> off_proc->idx1[curr_col];

                double dot_prod = 0.0;
                for (int l = prev_col_start; l<prev_col_end; l++)
                {
                    // here assume each aggregate block is dense hence 0 is stored explicitly
                    dot_prod += (T_csc -> off_proc->vals[l]) * (T_csc -> off_proc->vals[l - prev_col_start + curr_col_start]);
                }

                off_proc_norms[curr_col] = dot_prod;
            }
            comm->communicate_T(off_proc_norms, on_proc_norms, 1, func, func);

            // update upper-triangular entries of R
            for (int k = 0; k < n_aggs; k++)
            {
                R[R_flat_idx(k, j ,i)] = on_proc_norms[k*num_candidates +i];
            }

            // subtract the projections
            std::vector<double>& off_proc_dot = comm->communicate(on_proc_norms);
            // on_proc
            for (int k = 0; k< n_aggs; k++)
            {
                int prev_col = k*num_candidates+j;
                int curr_col = k*num_candidates+i;

                int prev_col_start = T_csc -> on_proc->idx1[prev_col];
                int prev_col_end = T_csc -> on_proc->idx1[prev_col +1];
                int curr_col_start = T_csc -> on_proc->idx1[curr_col];

                for (int l = prev_col_start; l<prev_col_end; l++)
                {
                    T_csc -> on_proc->vals[l-prev_col_start+curr_col_start] -=on_proc_norms[curr_col]* T_csc -> on_proc->vals[l];
                }
            }

            // off_proc
            for (int k = 0; k< off_proc_num_cols; k++)
            {
                int prev_col = k*num_candidates+j;
                int curr_col = k*num_candidates+i;

                int prev_col_start = T_csc -> off_proc->idx1[prev_col];
                int prev_col_end = T_csc -> off_proc->idx1[prev_col +1];
                int curr_col_start = T_csc -> off_proc->idx1[curr_col];

                for (int l = prev_col_start; l<prev_col_end; l++)
                {
                    T_csc -> off_proc->vals[l-prev_col_start+curr_col_start] -=off_proc_dot[curr_col]* T_csc -> off_proc->vals[l];
                }
            }
        }

        // normalizaton
        std::fill(on_proc_norms.begin(), on_proc_norms.end(), 0.0);
        std::fill(off_proc_norms.begin(), off_proc_norms.end(), 0.0);
        // on_proc
        for (int j = 0; j < n_aggs; j++)
        {
            int T_col = j*num_candidates+i;
            col_start = T_csc -> on_proc->idx1[T_col];
            col_end = T_csc -> on_proc->idx1[T_col+1];
            // add up the norms
            double norm2 = 0.0;
            for (int k = col_start; k< col_end; k++)
            {
                auto val = T_csc -> on_proc->vals[k];
                norm2 += val*val;
            }
            on_proc_norms[T_col] = norm2;
        }

        // off_proc
        for (int j = 0; j < off_proc_num_cols; j++)
        {
            int T_col = j*num_candidates+i;
            col_start = T_csc -> off_proc->idx1[T_col];
            col_end = T_csc -> off_proc->idx1[T_col+1];
            // add up the norms 
            double norm2 = 0.0;
            for (int k = col_start; k< col_end; k++)
            {
                auto val = T_csc -> off_proc->vals[k];
                norm2 += val*val;
            }
            off_proc_norms[T_col] = norm2;
        }

        comm->communicate_T(off_proc_norms, on_proc_norms, 1, func, func);

        // update R 
        for (int j = 0; j < n_aggs;j++)
        {
            int T_col = j*num_candidates+i;
            col_start = T_csc -> on_proc->idx1[T_col];
            col_end = T_csc -> on_proc->idx1[T_col+1];
            double norm = sqrt(on_proc_norms[T_col]);
            if (norm > threshold[j])
            {
                R[R_flat_idx(j, i, i)] = norm;
                scale = 1.0 / norm;
            }
            else 
            {
                R[R_flat_idx(j, i, i)] = 0;
                scale = 0.0;
            }

            on_proc_norms[T_col] = scale; // for off_proc comm

            for (int k = col_start; k< col_end; k++)
            {
                T_csc -> on_proc ->vals[k] *= scale;
            }
        }

        std::vector<double>& off_proc_scale = comm -> communicate(on_proc_norms);
        for (int j = 0; j < off_proc_num_cols; j++)
        {
            int T_col = j*num_candidates+i;
            col_start = T_csc -> off_proc->idx1[T_col];
            col_end = T_csc -> off_proc->idx1[T_col+1];
            for (int k = col_start; k< col_end; k++)
            {
                T_csc -> off_proc ->vals[k] *= off_proc_scale[T_col];
            }
        }

    }
 
    ParCSRMatrix* T = T_csc->to_ParCSR();
    delete T_csc;

    return T;
}


// BSR overload
static ParBSRMatrix* fit_candidates_bsr(ParBSRMatrix* A, const int n_aggs,
        const std::vector<int>& aggregates,
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm, double tol)
{
    BSRMatrix* A_on = static_cast<BSRMatrix*>(A->on_proc);
    BSRMatrix* A_off = static_cast<BSRMatrix*>(A->off_proc);
    assert(B.size() == A->local_num_rows*A_on->b_rows*num_candidates);

    int col_start, col_end, row;
    int global_col, local_col;
    double val, scale;
    CommPkg* comm;

    // Calculate off_proc_column_map and num off_proc (block) cols
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

    std::vector<int> on_proc_agg_marker(A->on_proc_num_cols, 0);
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
                on_proc_agg_marker[local_col] = 1;
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
    int global_num_aggs;
    RAPtor_MPI_Allreduce(&n_aggs, &global_num_aggs, 1, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    ParBSCMatrix* T_csc = new ParBSCMatrix(A->partition, A->global_num_rows, global_num_aggs,
            A->local_num_rows, n_aggs, off_proc_num_cols, A_on->b_rows, num_candidates);

    T_csc->off_proc_column_map = off_proc_column_map;
    T_csc->local_row_map = A->get_local_row_map();

    // Map on-proc aggregate roots to compact local aggregate indices
    // candidate-expanded T columns are created later
    std::vector<int> on_proc_agg_to_local_agg(A->on_proc_num_cols, -1);
    std::vector<int> on_proc_aggs;
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        if (on_proc_agg_marker[i])
        {
            on_proc_agg_to_local_agg[i] = on_proc_aggs.size();
            on_proc_aggs.emplace_back(A->on_proc_column_map[i]);
        }
    }
    for (int i = 0; i < AggOp_on->nnz; i++)
    {
        AggOp_on->idx2[i] = on_proc_agg_to_local_agg[AggOp_on->idx2[i]];
    }
    AggOp_on->n_cols = n_aggs;
    AggOp_off->n_cols = off_proc_num_cols;

    // expand on-proc columns with candidates for T_csc
    T_csc->on_proc_column_map.reserve(n_aggs);
    for (int i = 0; i < n_aggs; i++)
    {
        T_csc->on_proc_column_map.emplace_back(on_proc_aggs[i]);
    }


    // Convert AggOp matrices to CSC
    CSCMatrix* AggOp_on_csc = AggOp_on->to_CSC();
    CSCMatrix* AggOp_off_csc = AggOp_off->to_CSC();
    delete AggOp_on;
    delete AggOp_off;

    // Add columns of B to T (corresponding to pattern in AggOp)
    // Add on_process columns
    BSCMatrix* T_on = static_cast<BSCMatrix*>(T_csc->on_proc);
    BSCMatrix* T_off = static_cast<BSCMatrix*>(T_csc->off_proc);
    T_on->idx1[0] = 0;
    for (int i = 0; i < n_aggs; i++)
    {
        col_start = AggOp_on_csc->idx1[i];
        col_end = AggOp_on_csc->idx1[i+1];

        for (int col = col_start; col < col_end; col++)
        {
            row = AggOp_on_csc -> idx2[col];
            T_on->idx2.emplace_back(row);
            T_on->block_vals.emplace_back(new double[T_on -> b_size]());
            double* block = T_on->block_vals.back();
            // fill in the dense block
            for (int j = 0; j< num_candidates;j++)
            {
                for (int br = 0;br<T_on->b_rows;br++)
                {
                    // B is stored column-major
                    block[br*num_candidates+j] = B[j*(A->local_num_rows)*(A_on->b_rows)+row*(A_on->b_rows)+br];
                }
            }
        }
        T_on->idx1[i + 1] = T_on->idx2.size();

    }
    T_on->nnz = T_on->idx2.size();
    delete AggOp_on_csc;

    // Add off_process columns
    T_off->idx1[0] = 0;
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        col_start = AggOp_off_csc->idx1[i];
        col_end = AggOp_off_csc->idx1[i+1];

         for (int col = col_start; col < col_end; col++)
        {
            row = AggOp_off_csc -> idx2[col];
            T_off->idx2.emplace_back(row);
            T_off->block_vals.emplace_back(new double[T_off -> b_size]());
            double* block = T_off->block_vals.back();
            // fill in the dense block
            for (int j = 0; j< num_candidates;j++)
            {
                for (int br = 0;br<T_off->b_rows;br++)
                {
                    // B is stored column-major
                    block[br*num_candidates+j] = B[j*(A->local_num_rows)*(A_off->b_rows)+row*(A_off->b_rows)+br];
                }
            }
        }
        T_off->idx1[i + 1] = T_off->idx2.size();

    }
    T_off->nnz = T_off->idx2.size();
    delete AggOp_off_csc;

    // QR
    R.resize(n_aggs*num_candidates*num_candidates, 0.0);
    std::vector<double> on_proc_norms(n_aggs*num_candidates, 0);
    std::vector<double> off_proc_norms(off_proc_num_cols*num_candidates, 0);
    std::vector<double> threshold(n_aggs, 0);
    std::function<double(double, double)> func = &sum_func<double, double>;

    // helper to convert flat index for (R_agg)[i,j]
    auto R_flat_idx = [&](int agg_idx, int i, int j)
    {
        return agg_idx*num_candidates*num_candidates+ i*num_candidates+j;
    };

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

    for (int i = 0; i < num_candidates; i++)
    {
        std::fill(on_proc_norms.begin(), on_proc_norms.end(), 0.0);
        std::fill(off_proc_norms.begin(), off_proc_norms.end(), 0.0);

        // compute original norm of candidate i on_proc
        for (int j = 0; j < n_aggs; j++)
        {
            col_start = T_on->idx1[j];
            col_end = T_on->idx1[j+1];
            // add up the norms
            double norm2 = 0.0;
            for (int col = col_start; col< col_end; col++)
            {
                double* block = T_on->block_vals[col];
                for (int br = 0; br < T_on->b_rows; br++)
                {
                    norm2 += block[br*num_candidates+i]*block[br*num_candidates+i];
                }
            }
            on_proc_norms[j*num_candidates+i] = norm2;
        }

        // compute original norm of candidate i off_proc
        for (int j = 0; j < off_proc_num_cols; j++)
        {
            col_start = T_off->idx1[j];
            col_end = T_off->idx1[j+1];
            // add up the norms
            double norm2 = 0.0;
            for (int col = col_start; col< col_end; col++)
            {
                double* block = T_off->block_vals[col];
                 for (int br = 0; br < T_off->b_rows; br++)
                {
                    norm2 += block[br*num_candidates+i]*block[br*num_candidates+i];
                }
            }
            off_proc_norms[j*num_candidates+i] = norm2;
        }

        comm->communicate_T(off_proc_norms, on_proc_norms, num_candidates, func, func);

        for (int j = 0; j < n_aggs; j++)
        {
            threshold[j] = tol*sqrt(on_proc_norms[j*num_candidates+i]);
        }

        // orthogonalize against previous candidates
        for (int j = 0; j < i; j++)
        {
            std::fill(on_proc_norms.begin(), on_proc_norms.end(), 0.0);
            std::fill(off_proc_norms.begin(), off_proc_norms.end(), 0.0);
            // on_proc
            for (int k = 0; k< n_aggs; k++)
            {
                col_start = T_on->idx1[k];
                col_end = T_on->idx1[k+1];

                double dot_prod = 0.0;
                for (int col = col_start; col<col_end; col++)
                {
                    double* block = T_on-> block_vals[col];
                    for (int br = 0; br < T_on->b_rows; br++)
                    {
                        // current * prev
                        dot_prod += block[br*num_candidates+i]*block[br*num_candidates+j];
                    }

                }

                on_proc_norms[k*num_candidates+i] = dot_prod;
            }

            // off_proc
            for (int k = 0; k< off_proc_num_cols; k++)
            {
                col_start = T_off->idx1[k];
                col_end = T_off->idx1[k+1];

                double dot_prod = 0.0;
                for (int col = col_start; col<col_end; col++)
                {
                    double* block = T_off-> block_vals[col];
                    for (int br = 0; br < T_off->b_rows; br++)
                    {
                        // current * prev
                        dot_prod += block[br*num_candidates+i]*block[br*num_candidates+j];
                    }

                }

                off_proc_norms[k*num_candidates+i] = dot_prod;
            }
            comm->communicate_T(off_proc_norms, on_proc_norms, num_candidates, func, func);

            // update upper-triangular entries of R
            for (int k = 0; k < n_aggs; k++)
            {
                R[R_flat_idx(k, j ,i)] = on_proc_norms[k*num_candidates +i];
            }

            // subtract the projections
            std::vector<double>& off_proc_dot = comm->communicate(on_proc_norms, num_candidates);
            // on_proc
            for (int k = 0; k< n_aggs; k++)
            {
                col_start = T_on->idx1[k];
                col_end = T_on->idx1[k+1];

                for (int col = col_start; col<col_end; col++)
                {
                    double* block = T_on-> block_vals[col];
                    for (int br = 0; br < T_on->b_rows; br++)
                    {
                        block[br*num_candidates+i] -=on_proc_norms[k*num_candidates+i]* block[br*num_candidates+j];
                    }
                }
            }

            // off_proc
            for (int k = 0; k< off_proc_num_cols; k++)
            {
                col_start = T_off->idx1[k];
                col_end = T_off->idx1[k+1];

                for (int col = col_start; col<col_end; col++)
                {
                    double* block = T_off-> block_vals[col];
                    for (int br = 0; br < T_off->b_rows; br++)
                    {
                        block[br*num_candidates+i] -=off_proc_dot[k*num_candidates+i]* block[br*num_candidates+j];
                    }
                }
            }
        }

        // normalizaton
        std::fill(on_proc_norms.begin(), on_proc_norms.end(), 0.0);
        std::fill(off_proc_norms.begin(), off_proc_norms.end(), 0.0);
        // on_proc
        for (int j = 0; j < n_aggs; j++)
        {
            col_start = T_on->idx1[j];
            col_end = T_on->idx1[j+1];
            // add up the norms
            double norm2 = 0.0;
            for (int col = col_start; col< col_end; col++)
            {
                double* block = T_on->block_vals[col];
                for (int br = 0; br < T_on->b_rows; br++)
                {
                    norm2 += block[br*num_candidates+i]*block[br*num_candidates+i];
                }
            }
            on_proc_norms[j*num_candidates+i] = norm2;
        }

        // off_proc
        for (int j = 0; j < off_proc_num_cols; j++)
        {
            col_start = T_off->idx1[j];
            col_end = T_off->idx1[j+1];
            // add up the norms
            double norm2 = 0.0;
            for (int col = col_start; col< col_end; col++)
            {
                double* block = T_off->block_vals[col];
                for (int br = 0; br < T_off->b_rows; br++)
                {
                    norm2 += block[br*num_candidates+i]*block[br*num_candidates+i];
                }
            }
            off_proc_norms[j*num_candidates+i] = norm2;
        }

        comm->communicate_T(off_proc_norms, on_proc_norms, num_candidates, func, func);

        // update R
        for (int j = 0; j < n_aggs;j++)
        {
            int T_col = j*num_candidates+i;
            double norm = sqrt(on_proc_norms[T_col]);
            if (norm > threshold[j])
            {
                R[R_flat_idx(j, i, i)] = norm;
                scale = 1.0 / norm;
            }
            else
            {
                R[R_flat_idx(j, i, i)] = 0;
                scale = 0.0;
            }

            on_proc_norms[T_col] = scale; // for off_proc comm

            col_start = T_on->idx1[j];
            col_end = T_on->idx1[j+1];
            for (int col = col_start; col< col_end; col++)
            {
                double* block = T_on->block_vals[col];
                for (int br = 0; br < T_on->b_rows; br++)
                {
                    block[br*num_candidates+i] *= scale;
                }
            }
        }

        std::vector<double>& off_proc_scale = comm -> communicate(on_proc_norms, num_candidates);
        for (int j = 0; j < off_proc_num_cols; j++)
        {
            col_start = T_off->idx1[j];
            col_end = T_off->idx1[j+1];
            for (int col = col_start; col< col_end; col++)
            {
                double* block = T_off->block_vals[col];
                for (int br = 0; br < T_off->b_rows; br++)
                {
                    block[br*num_candidates+i] *= off_proc_scale[j*num_candidates+i];
                }
            }
        }

    }

    ParBSRMatrix* T = static_cast<ParBSRMatrix*>(T_csc->to_ParBSR());
    delete T_csc;

    return T;

}

template <class T, is_bsr_or_csr<T>>
T* fit_candidates(T* A,
        const int n_aggs, const std::vector<int>& aggregates,
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm, double tol)
{
    if constexpr (is_bsr_v<T>)
    {
        return fit_candidates_bsr(A, n_aggs, aggregates, B, R, num_candidates, tap_comm, tol);
    }
    else
    {
        return fit_candidates_csr(A, n_aggs, aggregates, B, R, num_candidates, tap_comm, tol);
    }
}

template ParCSRMatrix* fit_candidates(ParCSRMatrix*, const int,
        const std::vector<int>&, const std::vector<double>&,
        std::vector<double>&, int, bool, double);
template ParBSRMatrix* fit_candidates(ParBSRMatrix*, const int,
        const std::vector<int>&, const std::vector<double>&,
        std::vector<double>&, int, bool, double);

}//raptor namespace
