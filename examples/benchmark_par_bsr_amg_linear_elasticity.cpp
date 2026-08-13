// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "mfem.hpp"

#include "raptor/raptor.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int b_size = 1;
    int iter;

    ParCSRMatrix* A_csr = NULL;
    ParBSRMatrix* A = NULL;
    ParSmoothedAggregationSolver_T<ParBSRMatrix>* ml = NULL;
    ParVector x;
    ParVector b;
    std::vector<double> candidates;
    int num_candidates = 0;
    double t0, tfinal;

    const char* mesh_file = MFEM_MESH_DIR "/beam-tri.mesh";
    int order = 1;
    int ser_ref_levels = 1;
    int par_ref_levels = 1;
    int max_iter = 1000;
    double relax_damping = 0.5;
    double theta = 0.5;
    double tol = 1e-6;
    double material_contrast = 1.0;
    bool scale_spectral = true;
    bool static_cond = true;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order", "Finite element order.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                    "Number of serial uniform refinements.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                    "Number of parallel uniform refinements.");
    args.AddOption(&max_iter, "-mi", "--max-iter", "Maximum PCG iterations.");
    args.AddOption(&theta, "-t", "--theta", "Strength-of-connection threshold.");
    args.AddOption(&tol, "-tol", "--relative-tolerance",
                    "Relative standalone and PCG tolerance.");
    args.AddOption(&material_contrast, "-c", "--material-contrast",
                    "Ratio of material attribute 1 to the other attributes.");
    args.AddOption(&scale_spectral, "-sp", "--spectral-radius", "-no-sp",
                    "--no-spectral-radius", "Scale interpolation smoothing by spectral radius.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                    "--no-static-condensation", "Enable static condensation.");

    args.Parse();
    if (!args.Good())
    {
        if (rank == 0) args.PrintUsage(std::cout);
        MPI_Finalize();
        return 1;
    }

    A_csr = mfem_linear_elasticity(x, b, &b_size, mesh_file, order,
            ser_ref_levels, par_ref_levels, MPI_COMM_WORLD, &candidates,
            &num_candidates, static_cond, material_contrast);
    if (rank == 0) fprintf(stderr, "A has #of rows %d\n", 
                A_csr->global_num_rows);

    if ((A_csr->global_num_rows % b_size) ||
            (A_csr->global_num_cols % b_size) ||
            (A_csr->local_num_rows % b_size) ||
            (A_csr->on_proc_num_cols % b_size))
    {
        fprintf(stderr, "Matrix dimensions are not divisible by block size %d\n", b_size);

        delete A_csr;
        MPI_Finalize();
        return 1;
    }

    A = A_csr->to_ParBSR(b_size, b_size);

    if (rank == 0)
    {
        fprintf(stderr, "Converted CSR to BSR success!\n");
        fprintf(stderr, "A has #of block rows %d\n",
                A->global_num_rows);
        fprintf(stderr, "SA Solver on ParBSR matrix:\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ml = new ParSmoothedAggregationSolver_T<ParBSRMatrix>(
            theta, MIS, scale_spectral? BlockJacobiSpectralProlongation:BlockJacobiProlongation,
            Classical, BlockJacobi);
    ml->max_iterations = max_iter;
    ml->relax_weight = relax_damping;
    ml->solve_tol = tol;
    ml->num_variables = 1;
    ml->track_times = true;
    t0 = MPI_Wtime();
    ml->setup(A, candidates, num_candidates);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ml->print_hierarchy();
    ml->print_setup_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    ml->print_residuals(iter);
    ml->print_solve_times();

    MPI_Barrier(MPI_COMM_WORLD);
    ParVector pcg_sol = ParVector(x);
    std::vector<double> pcg_residuals;
    double precond_time = 0.0;
    double inner_product_time = 0.0;
    t0 = MPI_Wtime();
    PCG(A, ml, pcg_sol, b, pcg_residuals, ml->solve_tol,
            ml->max_iterations, &precond_time, &inner_product_time);
    tfinal = MPI_Wtime() - t0;

    double pcg_solve_time;
    double pcg_precond_time;
    double pcg_inner_product_time;
    MPI_Reduce(&tfinal, &pcg_solve_time, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&precond_time, &pcg_precond_time, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&inner_product_time, &pcg_inner_product_time, 1, MPI_DOUBLE,
            MPI_MAX, 0, MPI_COMM_WORLD);

    ParVector pcg_residual(b.global_n, b.local_n);
    A->residual(pcg_sol, b, pcg_residual);
    const double pcg_relative_residual = pcg_residual.norm(2) / b.norm(2);
    const int pcg_iterations = pcg_residuals.size() - 1;

    if (rank == 0)
    {
        printf("PCG Result:\n");
        printf("Iterations: %d\n", pcg_iterations);
        printf("Final Relative Residual: %e\n", pcg_relative_residual);
        printf("Total Solve Time: %e\n", pcg_solve_time);
        printf("AMG Preconditioner Time: %e\n", pcg_precond_time);
        printf("Inner Product Time: %e\n", pcg_inner_product_time);
    }
    delete ml;

    delete A;
    delete A_csr;

    MPI_Finalize();
    return 0;
}
