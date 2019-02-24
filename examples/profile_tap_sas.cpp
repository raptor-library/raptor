// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"

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

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

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

        strong_threshold = 0.5;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                strong_threshold = 0.0;
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
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols);
        b = ParVector(A->global_num_rows, A->local_num_rows);
        x.set_rand_values();
        A->mult(x, b);
        x.set_const_value(0.0);
    }
    ParVector tmp(A->global_num_rows, A->local_num_rows);

    int n_tests = 4;
    int n_iter = 100;

    ml = new ParSmoothedAggregationSolver(strong_threshold, MIS, JacobiProlongation, Symmetric, SOR);
    ml->max_iterations = 1000;
    ml->tap_amg = 0;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = false;
    ml->setup(A);

    aligned_vector<double> B(A->local_num_rows);
    aligned_vector<double> R;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        B[i] = 1.0;
    }

    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        if (rank == 0) printf("Level %d\n", i);
        ParLevel* level = ml->levels[i];
 

        /********************************
         * Profile Time to Create CommPkg
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            level->A->delete_comm(&(level->A->comm));
            level->A->delete_comm(&(level->A->tap_comm));
            level->A->delete_comm(&(level->A->tap_mat_comm));
            level->A->delete_comm(&(level->A->three_step));
            level->A->delete_comm(&(level->A->two_step));

            // Time to create ParComm (standard) for A
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            level->A->comm = new ParComm(level->A->partition, level->A->off_proc_column_map,
                    level->A->on_proc_column_map, 9243, MPI_COMM_WORLD);
            finalize_profile();
            print_profile("Form A ParComm");

            // Time to create TAP communicators for A
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            level->A->tap_comm = new TAPComm(level->A->partition, 
                level->A->off_proc_column_map, level->A->on_proc_column_map,
                false);
            finalize_profile();
            print_profile("Form A 2-Step TAPComm");
            level->A->delete_comm(&(level->A->tap_comm));

            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            level->A->tap_comm = new TAPComm(level->A->partition, 
                level->A->off_proc_column_map, level->A->on_proc_column_map,
                true);
            finalize_profile();
            print_profile("Form A 3-Step TAPComm");
            level->A->delete_comm(&(level->A->tap_comm));
        }

        level->A->delete_comm(&(level->A->comm));
        level->A->delete_comm(&(level->A->tap_comm));
        level->A->delete_comm(&(level->A->tap_mat_comm));
        level->A->delete_comm(&(level->A->two_step));
        level->A->delete_comm(&(level->A->three_step));

        level->P->delete_comm(&(level->P->comm));
        level->P->delete_comm(&(level->P->tap_comm));
        level->P->delete_comm(&(level->P->tap_mat_comm));
        level->P->delete_comm(&(level->P->two_step));
        level->P->delete_comm(&(level->P->three_step));

        level->A->comm = new ParComm(level->A->partition, level->A->off_proc_column_map,
                level->A->on_proc_column_map);
        level->A->two_step = new TAPComm(level->A->partition, level->A->off_proc_column_map,
                level->A->on_proc_column_map, false);
        level->A->three_step = new TAPComm(level->A->partition, level->A->off_proc_column_map,
                level->A->on_proc_column_map, true);
        level->A->set_tap_comm(level->A->three_step);
        level->A->set_tap_mat_comm(level->A->three_step);
        
        level->P->comm = new ParComm(level->P->partition, level->P->off_proc_column_map,
                level->P->on_proc_column_map);
        level->P->two_step = new TAPComm(level->P->partition, level->P->off_proc_column_map,
                level->P->on_proc_column_map, false);
        level->P->three_step = new TAPComm(level->P->partition, level->P->off_proc_column_map,
                level->P->on_proc_column_map, true);
        level->P->set_tap_comm(level->P->three_step);
        level->P->set_tap_mat_comm(level->P->three_step);


        /*********************************
         * Profile Time to Form Strength
         *********************************/
        ParCSRMatrix* S = level->A->strength(Classical, strong_threshold);
        for (int test = 0; test < n_tests; test++)
        {
            // Standard Comm
            delete S;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            S = level->A->strength(Symmetric, strong_threshold, false, 1, NULL);
            finalize_profile();
            print_profile("Strength");

            // Node-Aware Comm
            delete S;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            S = level->A->strength(Symmetric, strong_threshold, true, 1, NULL);
            finalize_profile();
            print_profile("TAP Strength");
        }



        /*********************************
         * Profile MIS Time
         *********************************/
        aligned_vector<int> states;
        aligned_vector<int> off_proc_states;
        for (int test = 0; test < n_tests; test++)
        {
            // Standard MIS
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            mis2(S, states, off_proc_states, false, NULL);
            finalize_profile();
            print_profile("MIS");

            // Node-Aware MIS
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            mis2(S, states, off_proc_states, true, NULL);
            finalize_profile();
            print_profile("TAP MIS");
        }


        /*********************************
         * Profile Aggregation Time
         *********************************/
        int n_aggs = 0;
        aligned_vector<int> aggregates;
        for (int test = 0; test < n_tests; test++)
        {
            // Standard Aggregation
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            n_aggs = aggregate(level->A, S, states, off_proc_states, aggregates, false,
                    NULL);
            finalize_profile();
            print_profile("Aggregation");
            aggregates.clear();


            // Node-Aware Aggregation
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            n_aggs = aggregate(level->A, S, states, off_proc_states, aggregates, true,
                    NULL);
            finalize_profile();
            print_profile("TAP Aggregation");
            aggregates.clear();
        }


        /*********************************
         * Profile Tentative Interp Time
         *********************************/
        n_aggs = aggregate(level->A, S, states, off_proc_states, aggregates, false,
                    NULL);
        double interp_tol = 1e-10;
        ParCSRMatrix* T = fit_candidates(level->A, n_aggs, aggregates, B, R, 
                    1, false, 1e-10);
        for (int test = 0; test < n_tests; test++)
        {
            // Standard Interpolation
            delete T;
            R.clear();
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            T = fit_candidates(level->A, n_aggs, aggregates, B, R, 1, false, 1e-10);
            finalize_profile();
            print_profile("Fit Candidates");

            // Node-Aware Interpolation
            delete T;
            R.clear();
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            T = fit_candidates(level->A, n_aggs, aggregates, B, R, 1, true, 1e-10);
            finalize_profile();
            print_profile("TAP Fit Candidates");
        }



        /*********************************
         * Profile Prolongation Smoothing
         *********************************/
        double prolong_weight = 3.0/4;
        int prolong_smooth_steps = 1;
        ParCSRMatrix* P = jacobi_prolongation(level->A, T, false, prolong_weight,
                prolong_smooth_steps);
        for (int test = 0; test < n_tests; test++)
        {
            // Standard Prolongation
            delete P;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            P = jacobi_prolongation(level->A, T, false, prolong_weight, prolong_smooth_steps);
            finalize_profile();
            print_profile("Jacobi Prolongation");

            // Node-Aware Prolongation
            delete P;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            P = jacobi_prolongation(level->A, T, true, prolong_weight, prolong_smooth_steps);
            finalize_profile();
            print_profile("TAP Jacobi Prolongation");
        }


        /*********************************
         * Profile Time to Create P Comm
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            P->delete_comm(&(P->comm));
            P->delete_comm(&(P->tap_comm));
            P->delete_comm(&(P->tap_mat_comm));
            P->delete_comm(&(P->three_step));
            P->delete_comm(&(P->two_step));

            // Time to create ParComm (standard) for P
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            P->comm = new ParComm(P->partition, P->off_proc_column_map,
                    P->on_proc_column_map);
            finalize_profile();
            print_profile("Form P ParComm");

            // Time to create TAP communicators for P
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map,
                    P->on_proc_column_map, false);
            P->init_tap_communicators(MPI_COMM_WORLD);
            finalize_profile();
            print_profile("Form P 2-Step TAPComm");
            P->delete_comm(&(P->tap_comm));

            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map,
                    P->on_proc_column_map, true);
            P->init_tap_communicators(MPI_COMM_WORLD);
            finalize_profile();
            print_profile("Form P 3-Step TAPComm");
            P->delete_comm(&(P->tap_comm));
        }
        P->delete_comm(&(P->comm));
        P->delete_comm(&(P->tap_comm));
        P->delete_comm(&(P->tap_mat_comm));
        P->delete_comm(&(P->two_step));
        P->delete_comm(&(P->three_step));
        P->comm = new ParComm(P->partition, P->off_proc_column_map, P->on_proc_column_map);
        P->two_step = new TAPComm(P->partition, P->off_proc_column_map,
                P->on_proc_column_map, false);
        P->three_step = new TAPComm(P->partition, P->off_proc_column_map,
                P->on_proc_column_map, true);
        P->set_tap_comm(P->three_step);
        P->set_tap_mat_comm(P->two_step);

        /*********************************
         * Profile Time to AP
         *********************************/
        ParCSRMatrix* AP;
        for (int test = 0; test < n_tests; test++)
        {
            // Standard AP
            AP = level->A->mult(P, false);
            delete AP;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            AP = level->A->mult(P, false);
            finalize_profile();
            print_profile("AP");
            delete AP;

            // Node-Aware AP
            AP = level->A->mult(P, true);
            delete AP;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            AP = level->A->mult(P, true);
            finalize_profile();
            print_profile("TAP AP");
            delete AP;
        }

        /*********************************
         * Profile Time to PTAP
         *********************************/
        AP = level->A->mult(P);
        ParCSRMatrix* Ac;
        for (int test = 0; test < n_tests; test++)
        {
            // Standard PTAP
            Ac = AP->mult_T(P, false);
            delete Ac;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            Ac = AP->mult_T(P, false);
            finalize_profile();
            print_profile("PTAP");
            delete Ac;

            // Node-Aware PTAP
            Ac = AP->mult_T(P, true);
            delete Ac;
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            Ac = AP->mult_T(P, true);
            finalize_profile();
            print_profile("TAP PTAP");
            delete Ac;
        }

        delete S;
        delete P;
        delete AP;

        std::copy(R.begin(), R.end(), B.begin());


        /*********************************
         * Profile Time to Relax
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, false);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, false);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("SOR");

            sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, true);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, true);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("TAP SOR");
        }

        /*********************************
         * Profile Residual Time
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            level->A->residual(level->x, level->b, level->tmp, false);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->A->residual(level->x, level->b, level->tmp, false);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("Residual");

            level->A->residual(level->x, level->b, level->tmp, true);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->A->residual(level->x, level->b, level->tmp, true);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("TAP Residual");
        }

        /*********************************
         * Profile P^Tr Time
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            level->P->mult_T(level->tmp, ml->levels[i+1]->b, false);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->P->mult_T(level->tmp, ml->levels[i+1]->b, false);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("P^Tr");

            level->P->mult_T(level->tmp, ml->levels[i+1]->b, true);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->P->mult_T(level->tmp, ml->levels[i+1]->b, true);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("TAP P^Tr");
        }

        /*********************************
         * Profile x += Pe time
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            level->P->mult_append(ml->levels[i+1]->x, level->x, false);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->P->mult_append(ml->levels[i+1]->x, level->x, false);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("x+=Pe");

            level->P->mult_append(ml->levels[i+1]->x, level->x, true);
            MPI_Barrier(MPI_COMM_WORLD);
            init_profile();
            for (int iter = 0; iter < n_iter; iter++)
            {
                level->P->mult_append(ml->levels[i+1]->x, level->x, true);
            }
            finalize_profile();
            average_profile(n_iter);
            print_profile("TAP x+=Pe");
        }
    }
    delete ml;

    MPI_Finalize();
    return 0;
}

