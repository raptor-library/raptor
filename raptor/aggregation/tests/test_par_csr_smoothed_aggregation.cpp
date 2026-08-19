// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "raptor/raptor.hpp"

using namespace raptor;

namespace
{
constexpr int num_candidates = 6;

std::vector<double> read_local_candidates(const char* filename,
        const Partition* partition)
{
    CSRMatrix* candidate_matrix = read_mm(filename);
    if (candidate_matrix == nullptr)
    {
        return {};
    }

    if (candidate_matrix->n_rows != partition->global_num_rows ||
            candidate_matrix->n_cols != num_candidates)
    {
        delete candidate_matrix;
        return {};
    }

    const int local_n = partition->local_num_rows;
    std::vector<double> candidates(local_n*num_candidates, 0.0);
    for (int local_row = 0; local_row < local_n; local_row++)
    {
        const int global_row = partition->first_local_row + local_row;
        for (int j = candidate_matrix->idx1[global_row];
                j < candidate_matrix->idx1[global_row + 1]; j++)
        {
            const int candidate = candidate_matrix->idx2[j];
            candidates[candidate*local_n + local_row] =
                    candidate_matrix->vals[j];
        }
    }

    delete candidate_matrix;
    return candidates;
}

}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

TEST(TestParCSRSmoothedAggregation, MultilevelElasticity)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const char* matrix_filename =
            "../../../../test_data/linear_elasticity_3D_A_csr.mtx";
    const char* candidate_filename =
            "../../../../test_data/linear_elasticity_3D_candidates.mtx";

    ParCSRMatrix* A = read_par_mm(matrix_filename);
    ASSERT_NE(A, nullptr);
    ASSERT_EQ(A->global_num_rows, 600);
    ASSERT_EQ(A->global_num_cols, 600);

    std::vector<double> candidates = read_local_candidates(
            candidate_filename, A->partition);
    ASSERT_EQ(candidates.size(),
            static_cast<size_t>(A->local_num_rows*num_candidates));

    const int global_n = A->global_num_rows;
    const int local_n = A->local_num_rows;
    const int first_local_row = A->partition->first_local_row;

    ParSmoothedAggregationSolver ml(
            0.5, MIS, JacobiProlongation, Classical, Jacobi);
    ml.relax_weight = 0.5;
    ml.max_levels = 5;
    ml.max_iterations = 500;
    ml.solve_tol = 1e-6;
    ml.track_times = true;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    ml.setup(A, candidates, num_candidates);
    const double local_setup_time = MPI_Wtime() - start;
    double setup_time = 0.0;
    MPI_Reduce(&local_setup_time, &setup_time, 1, MPI_DOUBLE, MPI_MAX,
            0, MPI_COMM_WORLD);

    const std::vector<ParLevel_T<ParBSRMatrix>*>& coarse_levels =
            ml.coarse_levels;

    if (rank == 0)
    {
        std::printf("CSR-BSR total AMG levels: %zu\n",
                ml.levels.size() + coarse_levels.size());
    }

    ASSERT_EQ(ml.levels.size(), 1);
    ASSERT_FALSE(coarse_levels.empty());
    ASSERT_NE(ml.levels[0]->P, nullptr);
    EXPECT_EQ(ml.levels[0]->A->on_proc->format(), CSR);
    EXPECT_EQ(ml.levels[0]->P->on_proc->format(), CSR);
    for (std::size_t level = 0; level < coarse_levels.size(); level++)
    {
        ASSERT_NE(coarse_levels[level]->A, nullptr);
        EXPECT_EQ(coarse_levels[level]->A->on_proc->format(), BSR);
        EXPECT_EQ(coarse_levels[level]->A->on_proc->b_rows,
                num_candidates);
        EXPECT_EQ(coarse_levels[level]->A->on_proc->b_cols,
                num_candidates);
        if (level + 1 < coarse_levels.size())
        {
            ASSERT_NE(coarse_levels[level]->P, nullptr);
            EXPECT_EQ(coarse_levels[level]->P->on_proc->format(), BSR);
        }
    }

    const int coarse_n = total_global_num_rows(*coarse_levels.back()->A);

    ParVector exact(global_n, local_n);
    ParVector solution(global_n, local_n);
    ParVector rhs(global_n, local_n);
    ParVector residual(global_n, local_n);
    for (int i = 0; i < local_n; i++)
    {
        const double global_row = first_local_row + i + 1.0;
        exact[i] = std::sin(0.37*global_row) +
                0.5*std::cos(0.11*global_row);
    }
    solution.set_const_value(0.0);
    A->mult(exact, rhs);

    const double initial_residual = rhs.norm(2);
    ASSERT_GT(initial_residual, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    const int iterations = ml.solve(solution, rhs);
    ml.print_residuals(iterations);
    const double local_solve_time = MPI_Wtime() - start;
    double solve_time = 0.0;
    MPI_Reduce(&local_solve_time, &solve_time, 1, MPI_DOUBLE,
            MPI_MAX, 0, MPI_COMM_WORLD);

    A->residual(solution, rhs, residual);
    const double relative_residual = residual.norm(2) / initial_residual;

    ParVector solution_error(solution);
    solution_error.axpy(exact, -1.0);
    const double relative_solution_error =
            solution_error.norm(2) / exact.norm(2);

    if (rank == 0)
    {
        std::printf(
        "CSR SA result:\n"
        "  ranks=%d\n"
        "  fine_size=%d\n"
        "  coarse_size=%d\n"
        "  iterations=%d\n"
        "  bsr_relative_residual=%.15e\n"
        "  relative_solution_error=%.15e\n"
        "  setup_time=%.6e\n"
        "  solve_time=%.6e\n",
        num_procs, global_n, coarse_n, iterations,
        relative_residual, relative_solution_error,
        setup_time, solve_time);
    }

    EXPECT_LT(coarse_n, global_n);
    EXPECT_LE(relative_residual, ml.solve_tol);
    EXPECT_LT(iterations, ml.max_iterations);

    delete A;
}
