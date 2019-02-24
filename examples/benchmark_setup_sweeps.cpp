// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"

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
        aligned_vector<int> grid;
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
        const char* file = "../../examples/LFAT5.pm";
        A = readParMatrix(file);
    }
    if (system != 2)
    {
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
 
    int n_tests = 10;
    int n_spmv_tests = 100;

    // Smoothed Aggregation AMG
    for (int s = 0; s < 3; s++)
    {
        if (rank == 0) printf("\n\nProlongation Smoothing Sweeps: %d:\n", s);
        ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, 
                Symmetric, SOR, s);
        ml->setup(A);
        for (int i = 0; i < ml->num_levels - 1; i++)
        {
            ParCSRMatrix* Al = ml->levels[i]->A;
            ParCSRMatrix* Pl = ml->levels[i]->P;
            Al->init_tap_communicators();
            Pl->init_tap_communicators();

            // Standard AP
            MPI_Barrier(MPI_COMM_WORLD);
            tfinal = 0;
            for (int j = 0; j < n_tests; j++)
            {
                t0 = MPI_Wtime();
                ParCSRMatrix* C = Al->mult(Pl);
                tfinal += MPI_Wtime() - t0;
                delete C;
            } 
            tfinal /= n_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Standard AP Time: %e\n", t0);

            // 3-step AP
            Al->comm_mat_type = NAP3;
            MPI_Barrier(MPI_COMM_WORLD);
            tfinal = 0;
            for (int j = 0; j < n_tests; j++)
            {
                t0 = MPI_Wtime();
                ParCSRMatrix* C = Al->tap_mult(Pl);
                tfinal += MPI_Wtime() - t0;
                delete C;
            } 
            tfinal /= n_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("3Step AP Time: %e\n", t0);

            // 2-step AP
            Al->comm_mat_type = NAP2;
            MPI_Barrier(MPI_COMM_WORLD);
            tfinal = 0;
            for (int j = 0; j < n_tests; j++)
            {
                t0 = MPI_Wtime();
                ParCSRMatrix* C = Al->tap_mult(Pl);
                tfinal += MPI_Wtime() - t0;
                delete C;
            } 
            tfinal /= n_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("2Step AP Time: %e\n", t0);

            ParCSRMatrix* AP = Al->mult(Pl);
            ParCSCMatrix* Pl_csc = Pl->to_ParCSC();
            // Standard PTAP
            MPI_Barrier(MPI_COMM_WORLD);
            tfinal = 0;
            for (int j = 0; j < n_tests; j++)
            {
                t0 = MPI_Wtime();
                ParCSRMatrix* C = AP->mult_T(Pl_csc);
                tfinal += MPI_Wtime() - t0;
                delete C;
            } 
            tfinal /= n_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Standard PTAP Time: %e\n", t0);

            // 3-step PTAP
            Al->comm_mat_type = NAP3;
            MPI_Barrier(MPI_COMM_WORLD);
            tfinal = 0;
            for (int j = 0; j < n_tests; j++)
            {
                t0 = MPI_Wtime();
                ParCSRMatrix* C = AP->tap_mult_T(Pl_csc);
                tfinal += MPI_Wtime() - t0;
                delete C;
            } 
            tfinal /= n_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("3Step PTAP Time: %e\n", t0);

            // 2-step PTAP
            Al->comm_mat_type = NAP2;
            MPI_Barrier(MPI_COMM_WORLD);
            tfinal = 0;
            for (int j = 0; j < n_tests; j++)
            {
                t0 = MPI_Wtime();
                ParCSRMatrix* C = AP->tap_mult_T(Pl_csc);
                tfinal += MPI_Wtime() - t0;
                delete C;
            } 
            tfinal /= n_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("2Step PTAP Time: %e\n", t0);
            delete Pl_csc;
            delete AP;


            // Standard Ax
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int j = 0; j < n_spmv_tests; j++)
            {
                Al->mult(ml->levels[i]->x, ml->levels[i]->b);
            } 
            tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Standard Ax Time: %e\n", t0);

            // 3-step Ax
            Al->comm_mat_type = NAP3;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int j = 0; j < n_spmv_tests; j++)
            {
                Al->tap_mult(ml->levels[i]->x, ml->levels[i]->b);
            } 
            tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("3Step Ax Time: %e\n", t0);

            // 2-step Ax
            Al->comm_mat_type = NAP2;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int j = 0; j < n_spmv_tests; j++)
            {
                Al->tap_mult(ml->levels[i]->x, ml->levels[i]->b);

            } 
            tfinal = (MPI_Wtime() - t0) / n_spmv_tests;
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("2Step Ax Time: %e\n", t0);

        }
        delete ml;
    }
    delete A;

    MPI_Finalize();
    return 0;
}

