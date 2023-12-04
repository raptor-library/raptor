// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor/raptor.hpp"

void form_hypre_weights(double** weight_ptr, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    double* weights;
    if (n_rows)
    {
        weights = new double[n_rows];
        for (int i = 0; i < n_rows; i++)
        {
            weights[i] = hypre_Rand();
        }
    }

    *weight_ptr = weights;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 5;
    int system = 0;
    double strong_threshold = 0.25;
    int iter;
    int num_variables = 1;

    coarsen_t coarsen_type = PMIS;
    interp_t interp_type = Extended;
    int hyp_coarsen_type = 8; // PMIS
    int hyp_interp_type = 6; // Extended
    int p_max_elmts = 0;
    int agg_num_levels = 0;


    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }
    if (system < 2)
    {
        int dim;
        double* stencil = NULL;
        std::vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            coarsen_type = Falgout;
            interp_type = ModClassical;
            hyp_coarsen_type = 6; // Falgout
            hyp_interp_type = 0; // Mod Classical

            dim = 2;
            grid.resize(dim, n);
            double eps = 0.001;
            double theta = M_PI/4.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
#ifdef USING_MFEM
    else if (system == 2)
    {
        const char* mesh_file = argv[2];
        int mfem_system = 0;
        int order = 2;
        int seq_refines = 1;
        int par_refines = 1;
        int max_dofs = 1000000;
        if (argc > 3)
        {
            mfem_system = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
                if (argc > 5)
                {
                    seq_refines = atoi(argv[5]);
                    max_dofs = atoi(argv[5]);
                    if (argc > 6)
                    {
                        par_refines = atoi(argv[6]);
                    }
                }
            }
        }

        coarsen_type = PMIS;
        interp_type = Extended;
        strong_threshold = 0.5;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                strong_threshold = 0.25;
                A = mfem_grad_div(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 2:
                strong_threshold = 0.9;
                A = mfem_linear_elasticity(x, b, &num_variables, mesh_file, order, 
                        seq_refines, par_refines);
                break;
            case 3:
                A = mfem_adaptive_laplacian(x, b, mesh_file, order);
                x.set_const_value(1.0);
                A->mult(x, b);
                x.set_const_value(0.0);
                break;
            case 4:
                A = mfem_dg_diffusion(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 5:
                A = mfem_dg_elasticity(x, b, &num_variables, mesh_file, order, seq_refines, par_refines);
                break;
        }
    }
#endif
    else if (system == 3)
    {
        if (argc > 2) A = readParMatrix(argv[2]);
	else A = readParMatrix("../../examples/LFAT5.pm");
    }
    if (system != 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols);
        b = ParVector(A->global_num_rows, A->local_num_rows);
        x.set_rand_values();
        A->mult(x, b);
        x.set_const_value(0.0);
    }

long nnz;
long local_nnz = A->local_nnz;
MPI_Reduce(&local_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
if (rank == 0) printf("A global n %d, nnz %lu\n", A->global_num_rows, nnz);

    // Create Hypre system
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(x);
    HYPRE_IJVector b_h_ij = convert(b);

    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);
    data_t* x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
    data_t* b_data = hypre_VectorData(hypre_ParVectorLocalVector(b_h));
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_data[i] = x[i];
        b_data[i] = b[i];
    }

    // Setup Hypre Hierarchy
    HYPRE_Solver solver_data;

    // Warm Up 
    solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
            hyp_coarsen_type, hyp_interp_type, p_max_elmts, agg_num_levels, 
            strong_threshold);
    hypre_BoomerAMGDestroy(solver_data);


    // Hypre Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
            hyp_coarsen_type, hyp_interp_type, p_max_elmts, agg_num_levels, 
            strong_threshold);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("HYPRE Setup Time: %e\n", t0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    HYPRE_BoomerAMGSolve(solver_data, A_h, b_h, x_h);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("HYPRE Solve Time: %e\n", t0);
    if (rank == 0) 
    {
        iter = hypre_ParAMGDataNumIterations((hypre_ParAMGData*) solver_data);
        printf("Solved in %d iterations\n", iter);
    }
    hypre_BoomerAMGDestroy(solver_data);


    // Warm Up
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->setup(A);
    ParVector xtmp = ParVector(x);
    ml->solve(xtmp, b);
    delete ml;

    // Ruge-Stuben AMG
    if (rank == 0) printf("Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = false;
    ml->store_residuals = false;
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    MPI_Barrier(MPI_COMM_WORLD);
    ParVector rss_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(rss_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    if (rank == 0) 
    {
        printf("Solved in %d iterations\n", iter);
    }
    delete ml;

    // TAP Ruge-Stuben AMG
    if (rank == 0) printf("\n\nTAP Ruge Stuben Solver: \n");
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = false;
    ml->store_residuals = false;
    ml->tap_amg = 3;
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    MPI_Barrier(MPI_COMM_WORLD);
    ParVector tap_rss_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(tap_rss_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    if (rank == 0) 
    {
        printf("Solved in %d iterations\n", iter);
    }
    delete ml;


    // Warm Up
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
            Symmetric, SOR);
    ml->setup(A);
    xtmp = ParVector(x);
    ml->solve(xtmp, b);
    delete ml;

    // Smoothed Aggregation AMG
    if (rank == 0) printf("\n\nSmoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, 
            Symmetric, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->track_times = false;
    ml->store_residuals = false;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ParVector sas_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(sas_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    if (rank == 0) 
    {
        printf("Solved in %d iterations\n", iter);
    }
    delete ml;

    // TAPSmoothed Aggregation AMG
    if (rank == 0) printf("\n\nTAP Smoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation,
            Symmetric, SOR);
    form_hypre_weights(&ml->weights, A->local_num_rows);
    ml->max_iterations = 1000;
    ml->solve_tol = 1e-07;
    ml->track_times = false;
    ml->store_residuals = false;
    ml->tap_amg = 3;
    t0 = MPI_Wtime();
    ml->setup(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Setup Time: %e\n", t0);
    ParVector tap_sas_sol = ParVector(x);
    t0 = MPI_Wtime();
    iter = ml->solve(tap_sas_sol, b);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Total Solve Time: %e\n", t0);
    if (rank == 0) 
    {
        printf("Solved in %d iterations\n", iter);
    }
    delete ml;

    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);

    delete A;

    MPI_Finalize();
    return 0;
}

