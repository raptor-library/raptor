// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"

#ifdef USING_MFEM
#include "external/mfem_wrapper.hpp"
#endif

using namespace raptor;


int main(int argc, char *argv[])
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
    std::vector<double> cache_array(10000);

    coarsen_t coarsen_type = PMIS;
    interp_t interp_type = Extended;

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
        if (argc > 2) file = argv[2];
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

    // Setup hierarchy first, so matrices are created for profiling
    ml = new ParRugeStubenSolver(strong_threshold, coarsen_type, interp_type);
    ml->tap_amg = 0;
    ml->setup(A);
    ml->print_hierarchy();

    std::vector<double> weights(A->local_num_rows);
    srand(time(NULL) + rank);
    for (int r = 0; r < A->local_num_rows; r++)
        weights[r] = rand() / RAND_MAX;


    // PROFILES SETUP AND SOLVE TIMES
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParVector& xl = ml->levels[i]->x;
        ParVector& bl = ml->levels[i]->b;
        ParVector& tmpl = ml->levels[i]->tmp;
        ParVector& bl1 = ml->levels[i+1]->b;
        ParVector& xl1 = ml->levels[i+1]->x;
        std::vector<int> states;
        std::vector<int> off_proc_states;

        if (!Al->tap_comm)
            Al->tap_comm = new TAPComm(Al->partition,
                    Al->off_proc_column_map, Al->on_proc_column_map);
        if (!Pl->tap_comm)
            Pl->tap_comm = new TAPComm(Pl->partition, 
                    Pl->off_proc_column_map, Pl->on_proc_column_map);
        if (!Al->tap_mat_comm)
            Al->tap_mat_comm = new TAPComm(Al->partition, 
                    Al->off_proc_column_map, Al->on_proc_column_map, false);
        if (!Pl->tap_mat_comm)
            Pl->tap_mat_comm = new TAPComm(Pl->partition,
                    Pl->off_proc_column_map, Pl->on_proc_column_map, false);
        if (!Al->comm)
            Al->comm = new ParComm(Al->partition, Al->off_proc_column_map, 
                    Al->on_proc_column_map);
        if (!Pl->comm)
            Pl->comm = new ParComm(Pl->partition, Pl->off_proc_column_map,
                    Pl->on_proc_column_map);

        ParCSRMatrix* Sl = Al->strength(Classical, strong_threshold);

        int n_times = 100;
        if (rank == 0) printf("Level %d\n", i);

        // TIME HMIS on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        split_hmis(Sl, states, off_proc_states, false, weights.data());
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("HMIS Time: %e\n", t0);

        // TIME TAP HMIS on Level i
        states.clear();
        off_proc_states.clear();
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        split_hmis(Sl, states, off_proc_states, true, weights.data());
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP HMIS Time: %e\n", t0);

        // TIME Interpolation on Level i
        ParCSRMatrix* Ptmp;
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        Ptmp = extended_interpolation(Al, Sl, states, off_proc_states, Al->comm);
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Interpolation Time: %e\n", t0);
        delete Ptmp;

        // TIME TAP Interpolation on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        Ptmp = extended_interpolation(Al, Sl, states, off_proc_states, Al->tap_comm);
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Interpolation Time: %e\n", t0);
        delete Ptmp;

        // TIME A*P on Level i
        ParCSRMatrix* APtmp;
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        APtmp = Al->mult(Pl);
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("A mult P Time: %e\n", t0);
        delete APtmp;

        // TIME TAP A*P on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        APtmp = Al->tap_mult(Pl);
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP A mult P Time: %e\n", t0);

        // TIME P^T*(AP) on Level i
        ParCSCMatrix* Pcsc = Pl->to_ParCSC();
        ParCSRMatrix* Actmp;
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        Actmp = APtmp->mult_T(Pcsc);
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("PT mult AT Time: %e\n", t0);
        delete Actmp;

        // TIME TAP P^T*(AP) on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        Actmp = APtmp->tap_mult_T(Pcsc);
        tfinal = (MPI_Wtime() - t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP PT mult AP Time: %e\n", t0);
        delete APtmp;
        delete Actmp;
        delete Pcsc;

        // TIME SOR on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        sor(Al, xl, bl, tmpl, n_times, 1.0);
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("SOR Time: %e\n", t0);
        
        // TIME TAP SOR on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        sor(Al, xl, bl, tmpl, n_times, 1.0, true);
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP SOR Time: %e\n", t0);

        // TIME Residual on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_times; i++)
        {
            Al->residual(xl, bl, tmpl);
        }
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Residual Time: %e\n", t0);

        // TIME TAP Residual on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_times; i++)
        {
            Al->tap_residual(xl, bl, tmpl);
        }
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Residual Time: %e\n", t0);

        // Time Restriction (P->mult_T(tmp, levels[i+1]->b))
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_times; i++)
        {
            Pl->mult_T(tmpl, bl1);
        }
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Restrict Time: %e\n", t0);

        // Time TAP Restriction (P->mult_T(tmp, levels[i+1]->b))
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_times; i++)
        {
            Pl->tap_mult_T(tmpl, bl1);
        }
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Restrict Time: %e\n", t0);

        // Time Interpolation (P->mult(levels[i+1]->x, tmp))
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_times; i++)
        {
            Pl->mult(xl1, tmpl);
        }
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Interp Time: %e\n", t0);

        // Time Interpolation (P->mult(levels[i+1]->x, tmp))
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_times; i++)
        {
            Pl->tap_mult(xl1, tmpl);
        }
        tfinal = (MPI_Wtime() - t0) / n_times;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Interp Time: %e\n", t0);

        delete Sl;
    }

    // Delete raptor hierarchy
    delete ml;
    delete A;

    MPI_Finalize();

    return 0;
}
