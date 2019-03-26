// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor.hpp"


void time_AP(ParCSRMatrix* A, ParCSRMatrix* P, int n_tests, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    ParCSRMatrix* C;

    // Warm-up
    C = A->mult(P, tap);
    delete C;

    MPI_Barrier(MPI_COMM_WORLD);
    tfinal = 0;
    for (int i = 0; i < n_tests; i++)
    {
        t0 = MPI_Wtime();
        C = A->mult(P, tap);
        tfinal += MPI_Wtime() - t0;
        delete C;
    }
    tfinal /= n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Time: %e\n", t0);
}


void time_PTAP(ParCSRMatrix* AP, ParCSCMatrix* P, int n_tests, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;
    ParCSRMatrix* C;

    // Warm-up
    C = AP->mult_T(P, tap);
    delete C;

    MPI_Barrier(MPI_COMM_WORLD);
    tfinal = 0;
    for (int i = 0; i < n_tests; i++)
    {
        t0 = MPI_Wtime();
        C = AP->mult_T(P, tap);
        tfinal += MPI_Wtime() - t0;
        delete C;
    }
    tfinal /= n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Time: %e\n", t0);
}

void time_Pe(ParCSRMatrix* P, ParVector& x, ParVector& b, int n_tests, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;

    // Warm-up
    P->mult(x, b, tap);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        P->mult(x, b, tap);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Time: %e\n", t0);
}

void time_PTr(ParCSRMatrix* P, ParVector& x, ParVector& b, int n_tests, bool tap)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0, tfinal;

    // Warm-up
    P->mult_T(x, b, tap);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        P->mult_T(x, b, tap);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Time: %e\n", t0);
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
   
    ml = new ParSmoothedAggregationSolver(strong_threshold);
    ml->setup(A);
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        Al->init_tap_communicators();
        Pl->init_tap_communicators();
  
        ParCSRMatrix* Pl2 = jacobi_prolongation(Al, Pl);
        ParCSRMatrix* AP = Al->mult(Pl);
        ParCSCMatrix* P_csc = Pl->to_ParCSC();
        ParCSCMatrix* P2_csc = Pl2->to_ParCSC();

        if (rank == 0) printf("Standard AP ");
        time_AP(Al, Pl, n_tests, false);
        if (rank == 0) printf("Standard AP2 ");
        time_AP(Al, Pl2, n_tests, false);

        Al->comm_mat_type = NAP2;
        Al->tap_mat_comm = Al->two_step;
        if (rank == 0) printf("Two Step AP ");
        time_AP(Al, Pl, n_tests, true);
        if (rank == 0) printf("Two Step AP2 ");
        time_AP(Al, Pl2, n_tests, true);

        Al->comm_mat_type = NAP3;
        Al->tap_mat_comm = Al->three_step;
        if (rank == 0) printf("Two Step AP ");
        time_AP(Al, Pl, n_tests, true);
        if (rank == 0) printf("Two Step AP2 ");
        time_AP(Al, Pl2, n_tests, true);



        if (rank == 0) printf("Standard PTAP ");
        time_PTAP(AP, P_csc, n_tests, false);
        if (rank == 0) printf("Standard PTAP2 ");
        time_PTAP(AP, P2_csc, n_tests, false);

        Al->comm_mat_type = NAP2;
        Al->tap_mat_comm = Al->two_step;
        if (rank == 0) printf("Two Step PTAP ");
        time_PTAP(AP, P_csc, n_tests, true);
        if (rank == 0) printf("Two Step PTAP2 ");
        time_PTAP(AP, P2_csc, n_tests, true);

        Al->comm_mat_type = NAP3;
        Al->tap_mat_comm = Al->three_step;
        if (rank == 0) printf("Two Step PTAP ");
        time_PTAP(AP, P_csc, n_tests, true);
        if (rank == 0) printf("Two Step PTAP2 ");
        time_PTAP(AP, P2_csc, n_tests, true);


        delete P_csc;
        delete P2_csc;
        delete AP;
        delete Pl2;
    }
    delete ml;

    delete A;

    MPI_Finalize();
    return 0;
}

