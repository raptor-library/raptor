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

    coarsen_t coarsen_type = HMIS;
    interp_t interp_type = Extended;

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;
    double comm_t, comm_mat_t;

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

        coarsen_type = HMIS;
        interp_type = Extended;
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

    int n_tests = 2;
    int n_iter = 100;

    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->max_iterations = 1000;
    ml->tap_amg = 0;
    ml->solve_tol = 1e-07;
    ml->num_variables = num_variables;
    ml->track_times = false;
    ml->setup(A);
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        if (rank == 0) printf("Level %d\n", i);
        ParLevel* level = ml->levels[i];
 

        /*********************************
         * Profile Time to Create CommPkg
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            if (level->A->comm) delete level->A->comm;
            if (level->A->tap_mat_comm) delete level->A->tap_mat_comm;
            if (level->A->tap_comm) delete level->A->tap_comm;

            // Time to create ParComm (standard) for A
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            level->A->comm = new ParComm(level->A->partition, level->A->off_proc_column_map,
                    level->A->on_proc_column_map, 9243, MPI_COMM_WORLD, &comm_t);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form A ParComm Time: %e\n", t0);
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form A ParComm Comm Time: %e\n", t0);

            // Time to create TAP communicators for A
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            level->A->init_tap_communicators(MPI_COMM_WORLD, &comm_t);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form A TAPComm Time: %e\n", t0);
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form A TAPComm Comm Time: %e\n", t0);

        }


        /*********************************
         * Profile Time to Form Strength
         *********************************/
        ParCSRMatrix* S = level->A->strength(Classical, strong_threshold);
        for (int test = 0; test < n_tests; test++)
        {
            // Standard Comm
            delete S;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            S = level->A->strength(Classical, strong_threshold, false, 1, NULL, &comm_t);
            tfinal = (MPI_Wtime() - t0);

            // Print Strength Time
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form Strength Time: %e\n", t0);


            // Node-Aware Comm
            delete S;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            S = level->A->strength(Classical, strong_threshold, true, 1, NULL, &comm_t);
            tfinal = (MPI_Wtime() - t0);

            // Print Strength Time
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Form Strength Time: %e\n", t0);
        }


        /*********************************
         * Profile Time to Split C/F
         *********************************/
        aligned_vector<int> states;
        aligned_vector<int> off_proc_states;
        for (int test = 0; test < n_tests; test++)
        {
            // Standard HMIS
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            split_hmis(S, states, off_proc_states, false, NULL, &comm_t);
            tfinal = (MPI_Wtime() - t0);

            // Print HMIS Time
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor HMIS Time: %e\n", t0);

            // Print HMIS Communication Time Corresponding to Max HMIS Time           
            if (fabs(t0 - tfinal) > zero_tol)
                comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor HMIS Comm Time: %e\n", t0);

            // Node-Aware HMIS
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            split_hmis(S, states, off_proc_states, true, NULL, &comm_t);
            tfinal = (MPI_Wtime() - t0);

            // Print HMIS Time
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP HMIS Time: %e\n", t0);

            // Print HMIS Communication Time Corresponding to Max HMIS Time           
            if (fabs(t0 - tfinal) > zero_tol)
                comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP HMIS Comm Time: %e\n", t0);

        }


        /*********************************
         * Profile Time to Form Interp
         *********************************/
        ParCSRMatrix* P = extended_interpolation(level->A, S, states, off_proc_states);
        for (int test = 0; test < n_tests; test++)
        {
            // Standard Interpolation
            delete P;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            comm_mat_t = 0;
            P = extended_interpolation(level->A, S, states, off_proc_states, false, 1, NULL,
                    &comm_t, &comm_mat_t);
            tfinal = (MPI_Wtime() - t0);

            // Print Interpolation Time
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Interpolation Time: %e\n", t0);

            // Print Interpolation Communication Time Corresponding to Max
            // Interpolation Time
            if (fabs(t0 - tfinal) > zero_tol)
            {
                comm_t = 0;
                comm_mat_t = 0;
            }
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Interpolation Comm Time: %e\n", t0);
            MPI_Reduce(&comm_mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Interpolation MATComm Time: %e\n", t0);


            // Node-Aware Interpolation
            delete P;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_t = 0;
            comm_mat_t = 0;
            P = extended_interpolation(level->A, S, states, off_proc_states, true, 1, NULL,
                    &comm_t, &comm_mat_t);
            tfinal = (MPI_Wtime() - t0);

            // Print Interpolation Time
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Interpolation Time: %e\n", t0);

            // Print Interpolation Communication Time Corresponding to Max
            // Interpolation Time
            if (fabs(t0 - tfinal) > zero_tol)
            {
                comm_t = 0;
                comm_mat_t = 0;
            }
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Interpolation Comm Time: %e\n", t0);
            MPI_Reduce(&comm_mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Interpolation MATComm Time: %e\n", t0);

        }


        /*********************************
         * Profile Time to Create P Comm
         *********************************/
        for (int test = 0; test < n_tests; test++)
        {
            delete P->comm;
            delete P->tap_mat_comm;
            delete P->tap_comm;

            // Time to create ParComm (standard) for P
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            P->comm = new ParComm(P->partition, P->off_proc_column_map,
                    P->on_proc_column_map);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form P ParComm Time: %e\n", t0);

            // Time to create TAP communicators for P
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            P->init_tap_communicators(MPI_COMM_WORLD);
            tfinal = (MPI_Wtime() - t0);
            MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Form P TAPComm Time: %e\n", t0);

        }


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
            t0 = MPI_Wtime();
            comm_mat_t = 0;
            AP = level->A->mult(P, false, &comm_mat_t);
            tfinal = (MPI_Wtime() - t0);
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor AP Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol)
                comm_mat_t = 0;
            MPI_Reduce(&comm_mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor AP Comm Time: %e\n", t0);
            delete AP;

            
            // Node-Aware AP
            AP = level->A->mult(P, true);
            delete AP;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_mat_t = 0;
            AP = level->A->mult(P, true, &comm_mat_t);
            tfinal = (MPI_Wtime() - t0);
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP AP Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol)
                comm_mat_t = 0;
            MPI_Reduce(&comm_mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP AP MATComm Time: %e\n", t0);
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
            Ac = AP->mult_T(P, false, &comm_mat_t);
            delete Ac;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_mat_t = 0;
            Ac = AP->mult_T(P, false, &comm_mat_t);
            tfinal = (MPI_Wtime() - t0);
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor PT(AP) Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol)
                comm_mat_t = 0;
            MPI_Reduce(&comm_mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor PT(AP) Comm Time: %e\n", t0);
            delete Ac;

            // Node-Aware PTAP
            Ac = AP->mult_T(P, true, &comm_mat_t);
            delete Ac;
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            comm_mat_t = 0;
            Ac = AP->mult_T(P, true, &comm_mat_t);
            tfinal = (MPI_Wtime() - t0);
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP PT(AP) Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol)
                comm_mat_t = 0;
            MPI_Reduce(&comm_mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP PT(AP) MATComm Time: %e\n", t0);
            delete Ac;
        }

        delete S;
        delete P;
        delete AP;

        /*********************************
         * Profile Time to Relax
         *********************************/
        sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4);        
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, false, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Relax Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Relax Comm Time: %e\n", t0);


            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                sor(level->A, level->x, level->b, level->tmp, 1, 3.0/4, true, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Relax Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Relax Comm Time: %e\n", t0);
        }

        /*********************************
         * Profile Residual Time
         *********************************/
        level->A->residual(level->x, level->b, level->tmp);
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                level->A->residual(level->x, level->b, level->tmp, false, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Residual Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor Residual Comm Time: %e\n", t0);

            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                level->A->residual(level->x, level->b, level->tmp, true, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Residual Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP Residual Comm Time: %e\n", t0);
        }

        /*********************************
         * Profile P^Tr Time
         *********************************/
        level->P->mult_T(level->tmp, ml->levels[i+1]->b);
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                level->P->mult_T(level->tmp, ml->levels[i+1]->b, false, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor P^T*r Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor P^T*r Comm Time: %e\n", t0);


            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                level->P->mult_T(level->tmp, ml->levels[i+1]->b, true, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP P^T*r Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP P^T*r Comm Time: %e\n", t0);
        }

        /*********************************
         * Profile x += Pe time
         *********************************/
        level->P->mult_append(ml->levels[i+1]->x, level->x);
        for (int test = 0; test < n_tests; test++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                level->P->mult_append(ml->levels[i+1]->x, level->x, false, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor x += P*e Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor x += P*e Comm Time: %e\n", t0);


            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            for (int iter = 0; iter < n_iter; iter++)
            {
                comm_t = 0;
                level->P->mult_append(ml->levels[i+1]->x, level->x, true, &comm_t);
            }
            tfinal = (MPI_Wtime() - t0) / n_iter;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP x += P*e Time: %e\n", t0);
            if (fabs(t0 - tfinal) > zero_tol) comm_t = 0;
            MPI_Reduce(&comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("RAPtor TAP x += P*e Comm Time: %e\n", t0);
        }
    }
    delete ml;

    MPI_Finalize();
    return 0;
}

