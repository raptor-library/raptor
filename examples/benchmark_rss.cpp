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
#include "ruge_stuben/par_ruge_stuben_solver.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "krylov/par_cg.hpp"

#ifdef USING_MFEM
  #include "gallery/external/mfem_wrapper.hpp"
#endif

//using namespace raptor;
//

void form_hypre_weights(aligned_vector<double>& weights, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hypre_SeedRand(2747 + rank);
    if (n_rows)
    {
        weights.resize(n_rows);
        for (int i = 0; i < n_rows; i++)
        {
            weights[i] = hypre_Rand();
        }
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x, x_tap;
    ParVector b;

    double t0;
    double raptor_setup, raptor_solve;
    double raptor_tap_solve;
    int num_variables = 1;

    int coarsen_type = 8;
    //int coarsen_type = 10; // HMIS (8 is PMIS)
    //int coarsen_type = 0; // CLJP
    //int coarsen_type = 6; // FALGOUT
    //int interp_type = 3; // Direct Interp
    //int interp_type = 0; // Classical Mod Interp
    int interp_type = 6;
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

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
    x_tap = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_tap[i] = x[i];
    }


    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    
    ParMultilevel* ml = new ParRugeStubenSolver(strong_threshold, HMIS, Extended, Classical, SOR);
    ml->num_variables = num_variables;
    ml->setup(A);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);


    // Solve Raptor Hierarchy
    aligned_vector<double> res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    //ml->tap_solve(x, b);
    //res = ml->get_residuals();
    PCG(A, ml, x, b, res, 1e-6, 100);
    raptor_solve = MPI_Wtime() - t0;
    clear_cache(cache_array);

    // TAP Solve Raptor
    aligned_vector<double> tap_res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    PCG(A, ml, x_tap, b, tap_res, 1e-6, 100);
    raptor_tap_solve = MPI_Wtime() - t0;
    clear_cache(cache_array);

    long lcl_nnz;
    long nnz;
    if (rank == 0) printf("Level\tNumRows\tNNZ\n");
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        lcl_nnz = Al->local_nnz;
        MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d\t%d\t%ld\n", i, Al->global_num_rows, nnz);
    }   

    if (rank == 0)
    {
        for (int i = 0; i < res.size(); i++)
        {
            printf("Res[%d] = %e\n", i+1, res[i]);
        }
    }   

    MPI_Reduce(&raptor_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Setup Time: %e\n", t0);
    MPI_Reduce(&raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Solve Time: %e\n", t0);
    MPI_Reduce(&raptor_tap_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor TAP Solve Time: %e\n", t0);
      
    // Delete raptor hierarchy
    delete ml;

    delete A;
    MPI_Finalize();

    return 0;
}

