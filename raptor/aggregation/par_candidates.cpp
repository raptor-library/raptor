// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_candidates.hpp"
#include <stdexcept>

namespace raptor {

template <class T, is_bsr_or_csr<T> = true>
static T* fit_candidates_helper(T* A,
        const int n_aggs, const std::vector<int>& aggregates,
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm, double tol)
{
    const int scalar_local_rows = total_local_num_rows(*A);
    if (B.size() != static_cast<size_t>(scalar_local_rows*num_candidates))
    {
        throw std::invalid_argument("candidate B size is inconsistent with A");
    }
    if (aggregates.size() != static_cast<size_t>(A->local_num_rows))
    {
        throw std::invalid_argument("aggregate size is inconsistent with A");
    }

    int rank, num_procs;
    RAPtor_MPI_Comm_rank(A->comm->mpi_comm, &rank);
    RAPtor_MPI_Comm_size(A->comm->mpi_comm, &num_procs);

    int block_rows = 1;
    if constexpr (is_bsr_v<T>)
    {
        block_rows = static_cast<BSRMatrix*>(A->on_proc)->b_rows;
    }

    // Calculate off_proc_column_map and num off_proc cols. These are
    // block columns for BSR and scalar columns for CSR.
    std::set<int> off_proc_col_set;
    for (int aggregate : aggregates)
    {
        if (aggregate >= 0 &&
                (aggregate < A->partition->first_local_col ||
                 aggregate > A->partition->last_local_col))
        {
            off_proc_col_set.insert(aggregate);
        }
    }

    std::vector<int> off_proc_column_map;
    std::map<int, int> global_to_local;
    for (int global_col : off_proc_col_set)
    {
        global_to_local[global_col] = off_proc_column_map.size();
        off_proc_column_map.emplace_back(global_col);
    }
    const int off_proc_num_cols = off_proc_column_map.size();

    // Create AggOp matrices
    std::vector<int> on_proc_agg_marker(A->on_proc_num_cols, 0);
    std::vector<int> on_proc_partition_to_col = A->map_partition_to_local();
    CSRMatrix* AggOp_on = new CSRMatrix(A->local_num_rows, -1);
    CSRMatrix* AggOp_off = new CSRMatrix(A->local_num_rows, -1);
    AggOp_on->idx1[0] = 0;
    AggOp_off->idx1[0] = 0;

    for (int i = 0; i < A->local_num_rows; i++)
    {
        const int global_col = aggregates[i];
        if (global_col >= 0)
        {
            if (global_col >= A->partition->first_local_col &&
                    global_col <= A->partition->last_local_col)
            {
                const int local_col = on_proc_partition_to_col[global_col - A->partition->first_local_col];
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

    // Map on-proc aggregate roots to compact local aggregate indices.
    // Candidate-expanded T columns are created later for CSR.
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

    int global_num_aggs;
    RAPtor_MPI_Allreduce(&n_aggs, &global_num_aggs, 1,
            RAPtor_MPI_INT, RAPtor_MPI_SUM, A->comm->mpi_comm);
    std::vector<int> num_aggs_per_proc(num_procs);
    RAPtor_MPI_Allgather(&n_aggs, 1, RAPtor_MPI_INT,
            num_aggs_per_proc.data(), 1, RAPtor_MPI_INT,
            A->comm->mpi_comm);
    
    // the compact global ID of the first agggregate owned by each rank
    int first_local_agg = 0;
    for (int i = 0; i < rank; i++)
    {
        first_local_agg += num_aggs_per_proc[i];
    }

    // Initialize tentative T.  CSR creates one scalar column per candidate
    // BSR creates one block column per aggregate
    int T_global_num_cols;
    int T_local_num_cols;
    int T_first_local_col;
    int T_off_proc_num_cols;
    // expand the candidates in T's partition
    if constexpr (is_bsr_v<T>)
    {
        T_global_num_cols = global_num_aggs;
        T_local_num_cols = n_aggs;
        T_first_local_col = first_local_agg;
        T_off_proc_num_cols = off_proc_num_cols;
    }
    else
    {
        T_global_num_cols = global_num_aggs*num_candidates;
        T_local_num_cols = n_aggs*num_candidates;
        T_first_local_col = first_local_agg*num_candidates;
        T_off_proc_num_cols = off_proc_num_cols*num_candidates;
    }

    Partition* T_partition = new Partition(A->global_num_rows,
            T_global_num_cols, A->local_num_rows, T_local_num_cols,
            A->partition->first_local_row, T_first_local_col,
            A->partition->topology);

    parallel_csc_matrix_t<T>* T_csc;
    if constexpr (is_bsr_v<T>)
    {
        T_csc = new ParBSCMatrix(T_partition,
                A->global_num_rows, T_global_num_cols,
                A->local_num_rows, T_local_num_cols,
                T_off_proc_num_cols, block_rows, num_candidates);
    }
    else
    {
        T_csc = new ParCSCMatrix(T_partition,
                A->global_num_rows, T_global_num_cols,
                A->local_num_rows, T_local_num_cols,
                T_off_proc_num_cols);
    }
    // T_csc is the sole owner of this newly allocated partition
    T_partition->num_shared = 0;

    // assign compact global ID for every aggregate on proc
    std::vector<int> on_proc_agg_global(n_aggs);
    for (int i = 0; i < n_aggs; i++)
    {
        on_proc_agg_global[i] = first_local_agg + i;
    }

    // communicate the compact global ID for off proc aggregate 
    ParComm agg_comm(A->partition, off_proc_column_map, on_proc_aggs,
            A->comm->key, A->comm->mpi_comm);
    std::vector<int>& off_proc_agg_global =
            agg_comm.communicate(on_proc_agg_global);

    // Expand on-proc and off-proc aggregate column maps 
    if constexpr (is_bsr_v<T>)
    {
        T_csc->on_proc_column_map = on_proc_agg_global;
        T_csc->off_proc_column_map = off_proc_agg_global;
    }
    else
    {
        T_csc->on_proc_column_map.reserve(n_aggs*num_candidates);
        for (int aggregate : on_proc_agg_global)
        {
            for (int candidate = 0; candidate < num_candidates; candidate++)
            {
                T_csc->on_proc_column_map.emplace_back(
                        aggregate*num_candidates + candidate);
            }
        }

        T_csc->off_proc_column_map.reserve(
                off_proc_num_cols*num_candidates);
        for (int aggregate : off_proc_agg_global)
        {
            for (int candidate = 0; candidate < num_candidates; candidate++)
            {
                T_csc->off_proc_column_map.emplace_back(
                        aggregate*num_candidates + candidate);
            }
        }
    }
    T_csc->local_row_map = A->get_local_row_map();

    // Convert AggOp matrices to CSC
    CSCMatrix* AggOp_on_csc = AggOp_on->to_CSC();
    CSCMatrix* AggOp_off_csc = AggOp_off->to_CSC();
    delete AggOp_on;
    delete AggOp_off;

    // helper to fill T corresponding to pattern in AggOp
    sequential_csc_matrix_t<T>* T_on =
            static_cast<sequential_csc_matrix_t<T>*>(T_csc->on_proc);
    sequential_csc_matrix_t<T>* T_off =
            static_cast<sequential_csc_matrix_t<T>*>(T_csc->off_proc);

    auto fill_T = [&](const CSCMatrix* AggOp,
            sequential_csc_matrix_t<T>* T_local, int num_aggs)
    {
        T_local->idx1[0] = 0;
        for (int aggregate = 0; aggregate < num_aggs; aggregate++)
        {
            const int start = AggOp->idx1[aggregate];
            const int end = AggOp->idx1[aggregate + 1];
            if constexpr (is_bsr_v<T>)
            {
                for (int i = start; i < end; i++)
                {
                    const int block_row = AggOp->idx2[i];
                    T_local->idx2.emplace_back(block_row);
                    T_local->block_vals.emplace_back(
                            new double[T_local->b_size]());
                    double* block = T_local->block_vals.back();
                    // Fill the dense block.  B is stored column-major by
                    // candidate, while each BSR block is stored row-major.
                    for (int candidate = 0;
                            candidate < num_candidates; candidate++)
                    {
                        for (int br = 0;
                                br < block_rows;
                                br++)
                        {
                            const int scalar_row =
                                    block_row*block_rows + br;
                            block[br*num_candidates + candidate] =
                                    B[candidate*scalar_local_rows + scalar_row];
                        }
                    }
                }
                T_local->idx1[aggregate + 1] = T_local->idx2.size();
            }
            else
            {
                for (int candidate = 0;
                        candidate < num_candidates; candidate++)
                {
                    const int T_col = aggregate*num_candidates + candidate;
                    for (int i = start; i < end; i++)
                    {
                        const int row = AggOp->idx2[i];
                        T_local->idx2.emplace_back(row);
                        // B is stored column-major by candidate.
                        T_local->vals.emplace_back(
                                B[candidate*scalar_local_rows + row]);
                    }
                    T_local->idx1[T_col + 1] = T_local->idx2.size();
                }
            }
        }
        T_local->nnz = T_local->idx2.size();
    };

    fill_T(AggOp_on_csc, T_on, n_aggs);
    fill_T(AggOp_off_csc, T_off, off_proc_num_cols);
    delete AggOp_on_csc;
    delete AggOp_off_csc;

    // helpers for dot product, axpy, and scaling 
    auto candidate_dot = [&](const sequential_csc_matrix_t<T>* matrix,
            int aggregate, int first_candidate, int second_candidate)
    {
        double dot = 0.0;
        if constexpr (is_bsr_v<T>)
        {
            const int start = matrix->idx1[aggregate];
            const int end = matrix->idx1[aggregate + 1];
            for (int i = start; i < end; i++)
            {
                const double* block = matrix->block_vals[i];
                for (int br = 0;
                        br < matrix->b_rows; br++)
                {
                    dot += block[br*num_candidates +
                            first_candidate] *
                            block[br*num_candidates +
                            second_candidate];
                }
            }
        }
        else
        {
            const int first_col =
                    aggregate*num_candidates + first_candidate;
            const int second_col =
                    aggregate*num_candidates + second_candidate;
            const int first_start = matrix->idx1[first_col];
            const int first_end = matrix->idx1[first_col + 1];
            const int second_start = matrix->idx1[second_col];
            // here assume each aggregate block is dense hence 0 is stored explicitly
            for (int i = first_start; i < first_end; i++)
            {
                dot += matrix->vals[i] *
                        matrix->vals[second_start + i - first_start];
            }
        }
        return dot;
    };

    auto candidate_axpy = [&](sequential_csc_matrix_t<T>* matrix,
            int aggregate, int destination_candidate,
            int source_candidate, double alpha)
    {
        if constexpr (is_bsr_v<T>)
        {
            const int start = matrix->idx1[aggregate];
            const int end = matrix->idx1[aggregate + 1];
            for (int i = start; i < end; i++)
            {
                double* block = matrix->block_vals[i];
                for (int br = 0;
                        br < matrix->b_rows; br++)
                {
                    block[br*num_candidates +
                            destination_candidate] += alpha *
                            block[br*num_candidates +
                            source_candidate];
                }
            }
        }
        else
        {
            const int destination_col =
                    aggregate*num_candidates + destination_candidate;
            const int source_col =
                    aggregate*num_candidates + source_candidate;
            const int destination_start = matrix->idx1[destination_col];
            const int source_start = matrix->idx1[source_col];
            const int source_end = matrix->idx1[source_col + 1];
            for (int i = source_start; i < source_end; i++)
            {
                matrix->vals[destination_start + i - source_start] +=
                        alpha*matrix->vals[i];
            }
        }
    };

    auto candidate_scale = [&](sequential_csc_matrix_t<T>* matrix,
            int aggregate, int candidate, double scale)
    {
        if constexpr (is_bsr_v<T>)
        {
            const int start = matrix->idx1[aggregate];
            const int end = matrix->idx1[aggregate + 1];
            for (int i = start; i < end; i++)
            {
                double* block = matrix->block_vals[i];
                for (int br = 0;
                        br < matrix->b_rows; br++)
                {
                    block[br*num_candidates + candidate] *= scale;
                }
            }
        }
        else
        {
            const int col = aggregate*num_candidates + candidate;
            for (int i = matrix->idx1[col];
                    i < matrix->idx1[col + 1]; i++)
            {
                matrix->vals[i] *= scale;
            }
        }
    };

    // QR
    R.assign(n_aggs*num_candidates*num_candidates, 0.0);
    std::vector<double> on_proc_work(n_aggs*num_candidates, 0.0);
    std::vector<double> off_proc_work(
            off_proc_num_cols*num_candidates, 0.0);
    std::vector<double> threshold(n_aggs, 0.0);
    std::function<double(double, double)> sum = &sum_func<double, double>;

    // Helper to convert (aggregate, row, column) into a flat R index.
    auto R_flat_idx = [&](int aggregate, int row, int col)
    {
        return aggregate*num_candidates*num_candidates +
                row*num_candidates + col;
    };

    // Create the communicator
    CommPkg* comm;
    if (tap_comm)
    {
        T_csc->tap_comm = new TAPComm(T_csc->partition,
                T_csc->off_proc_column_map, T_csc->on_proc_column_map,
                true, A->comm->mpi_comm);
        comm = T_csc->tap_comm;
    }
    else
    {
        T_csc->comm = new ParComm(T_csc->partition,
                T_csc->off_proc_column_map, T_csc->on_proc_column_map,
                A->comm->key, A->comm->mpi_comm);
        comm = T_csc->comm;
    }

    // CSR has one communication item per candidate-expanded scalar column;
    // BSR has one item per aggregate block with num_candidates values.
    const int comm_width = is_bsr_v<T> ? num_candidates : 1;

    for (int candidate = 0; candidate < num_candidates; candidate++)
    {
        // Compute the original norm of this candidate on the on-process and
        // off-process portions of every aggregate.
        std::fill(on_proc_work.begin(), on_proc_work.end(), 0.0);
        std::fill(off_proc_work.begin(), off_proc_work.end(), 0.0);
        for (int aggregate = 0; aggregate < n_aggs; aggregate++)
        {
            on_proc_work[aggregate*num_candidates + candidate] =
                    candidate_dot(T_on, aggregate, candidate, candidate);
        }
        for (int aggregate = 0;
                aggregate < off_proc_num_cols; aggregate++)
        {
            off_proc_work[aggregate*num_candidates + candidate] =
                    candidate_dot(T_off, aggregate, candidate, candidate);
        }
        comm->communicate_T(off_proc_work, on_proc_work,
                comm_width, sum, sum);

        for (int aggregate = 0; aggregate < n_aggs; aggregate++)
        {
            threshold[aggregate] = tol*std::sqrt(
                    on_proc_work[aggregate*num_candidates + candidate]);
        }

        // Orthogonalize this candidate against all previous candidates.
        for (int previous = 0; previous < candidate; previous++)
        {
            std::fill(on_proc_work.begin(), on_proc_work.end(), 0.0);
            std::fill(off_proc_work.begin(), off_proc_work.end(), 0.0);
            // Compute the local pieces of current dot previous.
            for (int aggregate = 0; aggregate < n_aggs; aggregate++)
            {
                on_proc_work[aggregate*num_candidates + candidate] =
                        candidate_dot(T_on, aggregate,
                                candidate, previous);
            }
            for (int aggregate = 0;
                    aggregate < off_proc_num_cols; aggregate++)
            {
                off_proc_work[aggregate*num_candidates + candidate] =
                        candidate_dot(T_off, aggregate,
                                candidate, previous);
            }
            comm->communicate_T(off_proc_work, on_proc_work,
                    comm_width, sum, sum);

            // Update the upper-triangular entries of R.
            for (int aggregate = 0; aggregate < n_aggs; aggregate++)
            {
                R[R_flat_idx(aggregate, previous, candidate)] =
                        on_proc_work[aggregate*num_candidates + candidate];
            }

            // Subtract the projections.
            std::vector<double>& off_proc_dot =
                    comm->communicate(on_proc_work, comm_width);
            for (int aggregate = 0; aggregate < n_aggs; aggregate++)
            {
                candidate_axpy(T_on, aggregate, candidate, previous,
                        -on_proc_work[aggregate*num_candidates + candidate]);
            }
            for (int aggregate = 0;
                    aggregate < off_proc_num_cols; aggregate++)
            {
                candidate_axpy(T_off, aggregate, candidate, previous,
                        -off_proc_dot[aggregate*num_candidates + candidate]);
            }
        }

        // normalization
        std::fill(on_proc_work.begin(), on_proc_work.end(), 0.0);
        std::fill(off_proc_work.begin(), off_proc_work.end(), 0.0);
        for (int aggregate = 0; aggregate < n_aggs; aggregate++)
        {
            on_proc_work[aggregate*num_candidates + candidate] =
                    candidate_dot(T_on, aggregate, candidate, candidate);
        }
        for (int aggregate = 0;
                aggregate < off_proc_num_cols; aggregate++)
        {
            off_proc_work[aggregate*num_candidates + candidate] =
                    candidate_dot(T_off, aggregate, candidate, candidate);
        }
        comm->communicate_T(off_proc_work, on_proc_work,
                comm_width, sum, sum);

        // Update the diagonal of R and scale the on-process candidate.
        for (int aggregate = 0; aggregate < n_aggs; aggregate++)
        {
            const int T_col = aggregate*num_candidates + candidate;
            const double norm = std::sqrt(on_proc_work[T_col]);
            double scale = 0.0;
            if (norm > threshold[aggregate])
            {
                R[R_flat_idx(aggregate, candidate, candidate)] = norm;
                scale = 1.0/norm;
            }
            on_proc_work[T_col] = scale;
            candidate_scale(T_on, aggregate, candidate, scale);
        }

        // Send each aggregate's normalization scale to its off-process.
        std::vector<double>& off_proc_scale =
                comm->communicate(on_proc_work, comm_width);
        for (int aggregate = 0;
                aggregate < off_proc_num_cols; aggregate++)
        {
            candidate_scale(T_off, aggregate, candidate,
                    off_proc_scale[aggregate*num_candidates + candidate]);
        }
    }

    T* tentative;
    if constexpr (is_bsr_v<T>)
    {
        tentative = static_cast<ParBSRMatrix*>(T_csc->to_ParBSR());
    }
    else
    {
        tentative = T_csc->to_ParCSR();
    }
    delete T_csc;
    return tentative;
}

template <class T, is_bsr_or_csr<T>>
T* fit_candidates(T* A,
        const int n_aggs, const std::vector<int>& aggregates,
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm, double tol)
{
    return fit_candidates_helper(A, n_aggs, aggregates, B, R,
            num_candidates, tap_comm, tol);
}

template ParCSRMatrix* fit_candidates(ParCSRMatrix*, const int,
        const std::vector<int>&, const std::vector<double>&,
        std::vector<double>&, int, bool, double);
template ParBSRMatrix* fit_candidates(ParBSRMatrix*, const int,
        const std::vector<int>&, const std::vector<double>&,
        std::vector<double>&, int, bool, double);

} // namespace raptor
