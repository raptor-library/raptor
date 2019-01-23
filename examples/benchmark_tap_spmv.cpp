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


#define eager_cutoff 1000
#define short_cutoff 62

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;
    int num_variables = 1;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;
    double raptor_setup, raptor_solve;

    double strong_threshold = 0.25;
    int cache_len = 10000;
    aligned_vector<double> cache_array(cache_len);
    aligned_vector<double> residuals;

    if (system < 2)
    {
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
            double eps = 0.1;
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
    if (system < 2)
    {
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
        if (argc > 3)
        {
            mfem_system = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
                if (argc > 5)
                {
                    seq_refines = atoi(argv[5]);
                    if (argc > 6)
                    {
                        par_refines = atoi(argv[6]);
                    }
                }
            }
        }
        
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
            case 5:
                A = mfem_dg_elasticity(x, b, &num_variables, mesh_file, order, seq_refines, par_refines);
        }                
    }
#endif

    else if (system == 3)
    {
        const char* file = "../../examples/LFAT5.mtx";
        int sym = 1;
        if (argc > 2)
        {
            file = argv[2];
            if (argc > 3)
            {
                sym = atoi(argv[3]);
            }
        }
        A = readParMatrix(file);
    }

    if (system != 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_rand_values();
        A->mult(x, b);
    }

    ParMultilevel* ml;
    clear_cache(cache_array);

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParRugeStubenSolver(strong_threshold, RS, Direct, Classical, SOR);
    ml->num_variables = num_variables;
    ml->setup(A);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);

    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParVector& xl = ml->levels[i]->x;
        ParVector& bl = ml->levels[i]->b;
        xl.set_rand_values();
        if (!Al->tap_comm) Al->tap_comm = new TAPComm(Al->partition,
                Al->off_proc_column_map, Al->on_proc_column_map);

        if (rank == 0) printf("Level %d\n", i);


        // Time SpMV on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < 100; i++)
        {
            Al->mult(xl, bl);
        }
        tfinal = (MPI_Wtime() - t0) / 100;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("SpMV Time: %e\n", t0);

        // Time TAPSpMV on Level i
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < 100; i++)
        {
            Al->tap_mult(xl, bl);
        }
        tfinal = (MPI_Wtime() - t0) / 100;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAPSpMV Time: %e\n", t0);

        // Gather number of messages / size (by inter/intra)
        int n_inter = 0;
        int n_intra = 0;
        int s_inter = 0;
        int s_intra = 0;
        int tap_n_inter = 0;
        int tap_n_intra = 0;
        int tap_s_inter = 0;
        int tap_s_intra = 0;
        int proc, node;
        int start, end;

        tap_n_inter = Al->tap_comm->global_par_comm->send_data->num_msgs;
        tap_s_inter = Al->tap_comm->global_par_comm->send_data->size_msgs;
        tap_n_intra = Al->tap_comm->local_L_par_comm->send_data->num_msgs 
            + Al->tap_comm->local_S_par_comm->send_data->num_msgs
            + Al->tap_comm->local_R_par_comm->send_data->num_msgs;
        tap_s_intra = Al->tap_comm->local_L_par_comm->send_data->size_msgs 
            + Al->tap_comm->local_S_par_comm->send_data->size_msgs
            + Al->tap_comm->local_R_par_comm->send_data->size_msgs;

        int rank_node = Al->partition->topology->get_node(rank);
        for (int i = 0; i < Al->comm->send_data->num_msgs; i++)
        {
            proc = Al->comm->send_data->procs[i];
            start = Al->comm->send_data->indptr[i];
            end = Al->comm->send_data->indptr[i+1];
            node = Al->partition->topology->get_node(proc);
            if (node == rank_node)
            {
                n_intra++;
                s_intra += (end - start);
            }
            else
            {
                n_inter++;
                s_inter += (end - start);
            }
        }

        int comm;
        MPI_Reduce(&n_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Intra Msgs: %d\n", comm);
        MPI_Reduce(&n_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Inter Msgs: %d\n", comm);
        MPI_Reduce(&s_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Size Intra Msgs: %d\n", comm);
        MPI_Reduce(&s_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Size Inter Msgs: %d\n", comm);
        MPI_Reduce(&tap_n_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Num Intra Msgs: %d\n", comm);
        MPI_Reduce(&tap_n_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Num Inter Msgs: %d\n", comm);
        MPI_Reduce(&tap_s_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Size Intra Msgs: %d\n", comm);
        MPI_Reduce(&tap_s_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Size Inter Msgs: %d\n", comm);
    }


    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}

