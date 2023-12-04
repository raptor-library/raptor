// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor/core/par_matrix.hpp"
#include "raptor/core/par_vector.hpp"
#include "raptor/core/types.hpp"
#include "raptor/gallery/par_stencil.hpp"
#include "raptor/gallery/laplacian27pt.hpp"
#include "raptor/gallery/diffusion.hpp"
#include "raptor/gallery/par_matrix_IO.hpp"
#include "raptor/multilevel/par_multilevel.hpp"
#include "raptor/ruge_stuben/par_ruge_stuben_solver.hpp"

#ifdef USING_MFEM
#include "raptor/external/mfem_wrapper.hpp"
#endif

#define eager_cutoff 1000
#define short_cutoff 62

using namespace raptor;

void print_times(double time, double time_comm, const char* name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0;
    MPI_Reduce(&time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time: %e\n", name, t0);
    MPI_Reduce(&time_comm, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time Comm: %e\n", name, t0);
}

void print_tap_times(double time, double time_comm, const char* name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    double t0;
    MPI_Reduce(&time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s TAP Time: %e\n", name, t0);
    MPI_Allreduce(&time_comm, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("%s TAP Comm Time: %e\n", name, t0);
}

void time_spgemm(ParCSRMatrix* A, ParCSRMatrix* P, bool tap)
{
    if (!A->comm) A->comm = new ParComm(A->partition, 
            A->off_proc_column_map, A->on_proc_column_map);

    int n_tests = 10;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);

    init_profile();
    for (int i = 1; i < n_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->mult(P, tap);
        delete C;
    }
    finalize_profile();
    average_profile(n_tests);
    double tfinal = total_t / n_tests;
    double comm_t = p2p_t / n_tests;

    if (tap)
    {
        print_tap_times(tfinal, comm_t, "SpGEMM");
    }
    else
    {
        print_times(tfinal, comm_t, "SpGEMM");
        //A->print_mult(P);
    }
}

void time_spgemm_T(ParCSRMatrix* A, ParCSCMatrix* P, bool tap)
{
    if (!P->comm) P->comm = new ParComm(P->partition, 
            P->off_proc_column_map, P->on_proc_column_map);

    int n_tests = 10;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);

    // Time SpGEMM on Level i
    double tfinal;
    double comm_t = 0;

    init_profile();
    for (int i = 1; i < n_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->mult_T(P, tap);
        delete C;
    }
    finalize_profile();
    average_profile(n_tests);
    tfinal = total_t / n_tests;
    comm_t = p2p_t / n_tests;

    if (tap)
    {
        print_tap_times(tfinal, comm_t, "Transpose SpGEMM");
    }
    else
    {
        print_times(tfinal, comm_t, "Transpose SpGEMM");
        //A->print_mult_T(P);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim = 3;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A = NULL;
    ParVector x;
    ParVector b;

    int num_variables = 1;
    relax_t relax_type = SOR;
    coarsen_t coarsen_type = CLJP;
    interp_t interp_type = ModClassical;
    double strong_threshold = 0.25;

    std::vector<double> residuals;

    if (system < 2)
    {
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
        strong_threshold = 0.0;
        switch (mfem_system)
        {
            case 0:
                A = mfem_laplacian(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 1:
                A = mfem_grad_div(x, b, mesh_file, order, seq_refines, par_refines);
                break;
            case 2:
                strong_threshold = 0.5;
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
    }

    ParMultilevel* ml;

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type, Classical, relax_type);
    ml->num_variables = num_variables;
    ml->setup(A);


    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParCSCMatrix* Pl_csc = Pl->to_ParCSC();
        ParCSRMatrix* AP = Al->mult(Pl);

        int A_nnz, P_nnz;
        MPI_Reduce(&Al->local_nnz, &A_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&Pl->local_nnz, &P_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("Level %d\n", i);
            printf("A %d x %d, %d nnz\n", Al->global_num_rows, Al->global_num_cols, A_nnz);
            printf("P %d x %d, %d nnz\n", Pl->global_num_rows, Pl->global_num_cols, P_nnz);
        }
        if (rank == 0) printf("A*P:\n");
        time_spgemm(Al, Pl, false);
        
        if (rank == 0) printf("\nTAP A*P:\n");
        time_spgemm(Al, Pl, true);

        if (rank == 0) printf("\nSimple TAP A*P:\n");
        time_spgemm(Al, Pl, true);

        if (rank == 0) printf("\nP.T*AP:\n");
        time_spgemm_T(AP, Pl_csc, false);

        if (rank == 0) printf("\nTAP P.T*AP:\n");
        time_spgemm_T(AP, Pl_csc, true);

        if (rank == 0) printf("\nSimple TAP P.T*AP:\n");
        time_spgemm_T(AP, Pl_csc, true);

        delete Pl_csc;
        delete AP;

        if (rank == 0) printf("Now testing P_new...\n");
        ParCSRMatrix* P_new = Al->mult(Pl);
        ParCSRMatrix* AP_new = Al->mult(P_new);
        ParCSCMatrix* P_new_csc = P_new->to_ParCSC();
        MPI_Reduce(&P_new->local_nnz, &P_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("Pnew %d x %d, %d nnz\n", P_new->global_num_rows, P_new->global_num_cols, P_nnz);
        }

        if (rank == 0) printf("A*P:\n");
        time_spgemm(Al, P_new, false);
        
        if (rank == 0) printf("\nTAP A*P:\n");
        time_spgemm(Al, P_new, true);

        if (rank == 0) printf("\nSimple TAP A*P:\n");
        time_spgemm(Al, P_new, true);

        if (rank == 0) printf("\nP.T*AP:\n");
        time_spgemm_T(AP_new, P_new_csc, false);

        if (rank == 0) printf("\nTAP P.T*AP:\n");
        time_spgemm_T(AP_new, P_new_csc, true);        

        if (rank == 0) printf("\nSimple TAP P.T*AP:\n");
        time_spgemm_T(AP_new, P_new_csc, true);

        delete P_new;
        delete P_new_csc;
        delete AP_new;
    }

    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}


