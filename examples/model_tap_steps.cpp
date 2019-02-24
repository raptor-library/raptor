// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"


void time_mult(ParCSRMatrix* A, ParCSRMatrix* P, int n_tests, bool tap, const char* name)
{
    ParCSRMatrix* AP = A->mult(P, tap);
    delete AP;
    for (int test = 0; test < n_tests; test++)
    {
        init_profile();
        AP = A->mult(P, tap);
        finalize_profile();
        print_profile(name);
        delete AP;
    }
}

void time_mult_T(ParCSRMatrix* AP, ParCSRMatrix* P, int n_tests, bool tap, const char* name)
{
    ParCSRMatrix* PTAP = AP->mult_T(P, tap);
    delete PTAP;
    for (int test = 0; test < n_tests; test++)
    {
	init_profile();
        PTAP = AP->mult_T(P, tap);
	finalize_profile();
	print_profile(name);
        delete PTAP;
    }
}

void time_mult(ParCSRMatrix* A, ParVector& x, ParVector& b, int n_tests, bool tap, const char* name)
{
    int n_iter = 100;
    A->mult(x, b, tap);
    for (int test = 0; test < n_tests; test++)
    {
        init_profile();
        for (int iter = 0; iter < n_iter; iter++)
        {
            A->mult(x, b, tap);
        }
        finalize_profile();
        average_profile(n_iter);
        print_profile(name);
    }
}

void time_mult_T(ParCSRMatrix* A, ParVector& x, ParVector& b, int n_tests, bool tap, const char* name)
{
    int n_iter = 100;
    A->mult_T(x, b, tap);
    for (int test = 0; test < n_tests; test++)
    {
        init_profile();
        for (int iter = 0; iter < n_iter; iter++)
        {
            A->mult_T(x, b, tap);
        }
        finalize_profile();
        average_profile(n_iter);
        print_profile(name);
    }
}

void time_steps(ParMultilevel* ml)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_tests = 5; 
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        if (rank == 0) printf("Level %d\n", i);
        ParLevel* level = ml->levels[i];

        TAPComm* tap_two_comm;
        TAPComm* tap_three_comm;

        if (!level->A->comm) 
            level->A->comm = new ParComm(level->A->partition, level->A->off_proc_column_map,
                    level->A->on_proc_column_map);
        if (!level->A->tap_comm) 
            level->A->tap_comm = new TAPComm(level->A->partition, level->A->off_proc_column_map,
                    level->A->on_proc_column_map);
        if (!level->A->tap_mat_comm) 
            level->A->tap_mat_comm = new TAPComm(level->A->partition, level->A->off_proc_column_map,
                    level->A->on_proc_column_map, false);

        if (!level->P->comm) 
            level->P->comm = new ParComm(level->P->partition, level->P->off_proc_column_map,
                    level->P->on_proc_column_map);
        if (!level->P->tap_comm) 
            level->P->tap_comm = new TAPComm(level->P->partition, level->P->off_proc_column_map,
                    level->P->on_proc_column_map);
        if (!level->P->tap_mat_comm) 
            evel->P->tap_mat_comm = new TAPComm(level->P->partition, level->P->off_proc_column_map,
                    level->P->on_proc_column_map, false);

        ParCSRMatrix* Al = level->A;
        ParCSRMatrix* Pl = level->P;
        ParVector& xl = level->x;
        ParVector& bl = level->b;
        ParVector& xlp1 = ml->levels[i+1]->x;

        /*********************************
         * Profile Time to AP
         *********************************/
        if (rank == 0) printf("AP\n");
        Al->print_mult(Pl);
        tap_two_comm = Al->tap_mat_comm;
        tap_three_comm = Al->tap_comm;
        time_mult(Al, Pl, n_tests, false, "Standard");
        Al->tap_mat_comm = tap_two_comm;
        time_mult(Al, Pl, n_tests, true, "Two-Step");
        Al->tap_mat_comm = tap_three_comm;
        time_mult(Al, Pl, n_tests, true, "Three-Step");
        Al->tap_mat_comm = tap_two_comm;
        
        /*********************************
         * Profile Time to PTAP
         *********************************/
        if (rank == 0) printf("PTAP\n");
        ParCSRMatrix* AP = Al->mult(Pl);
        ParCSCMatrix* P_csc = Pl->to_ParCSC();
        AP->print_mult_T(P_csc);
        tap_two_comm = Pl->tap_mat_comm;
        tap_three_comm = Pl->tap_comm;
        time_mult_T(AP, Pl, n_tests, false, "Standard");
        Pl->tap_mat_comm = tap_two_comm;
        time_mult_T(AP, Pl, n_tests, true, "Two-Step");
        Pl->tap_mat_comm = tap_three_comm;
        time_mult_T(AP, Pl, n_tests, true, "Three-Step");
        Pl->tap_mat_comm = tap_two_comm;
        delete P_csc;
        delete AP;
        
        /*********************************
         * Profile Time to Residal
         *********************************/
        if (rank == 0) printf("Ax\n");
        Al->print_mult();
        tap_two_comm = Al->tap_mat_comm;
        tap_three_comm = Al->tap_comm;
        time_mult(Al, xl, bl, n_tests, false, "Standard");
        Al->tap_comm = tap_two_comm;
        time_mult(Al, xl, bl, n_tests, true, "Two-Step");
        Al->tap_comm = tap_three_comm;
        time_mult(Al, xl, bl, n_tests, true, "Three-Step");
        Al->tap_comm = tap_three_comm;

        /*********************************
         * Profile Time P^T*r
         *********************************/
        if (rank == 0) printf("P^Tr\n");
        Pl->print_mult_T();
        tap_two_comm = Pl->tap_mat_comm;
        tap_three_comm = Pl->tap_comm;
        time_mult_T(Pl, xl, xlp1, n_tests, false, "Standard");
        Pl->tap_comm = tap_two_comm;
        time_mult_T(Pl, xl, xlp1, n_tests, true, "Two-Step");
        Pl->tap_comm = tap_three_comm;
        time_mult_T(Pl, xl, xlp1, n_tests, true, "Three-Step");
        Pl->tap_comm = tap_three_comm;


        /*********************************
         * Profile Time to Residal
         *********************************/
        if (rank == 0) printf("Pe\n");
        Pl->print_mult();
        tap_two_comm = Pl->tap_mat_comm;
        tap_three_comm = Pl->tap_comm;
        time_mult(Pl, xlp1, bl, n_tests, false, "Standard");
        Pl->tap_comm = tap_two_comm;
        time_mult(Pl, xlp1, bl, n_tests, true, "Two-Step");
        Pl->tap_comm = tap_three_comm;
        time_mult(Pl, xlp1, bl, n_tests, true, "Three-Step");
        Pl->tap_comm = tap_three_comm;

    }
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

    coarsen_t coarsen_type = HMIS;
    interp_t interp_type = Extended;

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

    int n_tests = 4;
    int n_iter = 100;

    if (rank == 0) printf("Ruge Stuben Solver:\n");
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, SOR);
    ml->tap_amg = 0;
    ml->track_times = false;
    ml->setup(A);
    time_steps(ml);
    delete ml;

    if (rank == 0) printf("\n\nSmoothed Aggregation Solver:\n");
    ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->tap_amg = 0;
    ml->track_times = false;
    ml->setup(A);
    time_steps(ml);
    delete ml;

    MPI_Finalize();
    return 0;
}

