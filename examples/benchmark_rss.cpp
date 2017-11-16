// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
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
#include "tests/hypre_compare.hpp"
#include "gallery/external/hypre_wrapper.hpp"

#ifdef USING_MFEM
#include "gallery/external/mfem_wrapper.hpp"
#endif

//using namespace raptor;
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
    ParVector x;
    ParVector b;

    double t0;
    double hypre_setup, hypre_solve;
    double raptor_setup, raptor_solve;

    int coarsen_type = 0; // CLJP
    //int coarsen_type = 6; // FALGOUT
    //int interp_type = 3; // Direct Interp
    int interp_type = 0; // Classical Mod Interp
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

    int cache_len = 10000;
    int num_tests = 2;

    std::vector<double> cache_array(cache_len);

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
            double theta = M_PI/8.0;
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
        char* mesh_file = argv[2];
        int num_elements = 2;
        int order = 3;
        if (argc > 3)
        {
            num_elements = atoi(argv[3]);
            if (argc > 4)
            {
                order = atoi(argv[4]);
            }
        }
        A = mfem_linear_elasticity(x, b, mesh_file, num_elements, order);
    }
#endif
    else if (system == 3)
    {
        char* file = "../../examples/LFAT5.mtx";
        int sym = 1;
        if (argc > 2)
        {
            file = argv[2];
            if (argc > 3)
            {
                sym = atoi(argv[3]);
            }
        }
        A = readParMatrix(file, MPI_COMM_WORLD, 1, sym);
    }

    if (system != 2)
    {
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_const_value(1.0);
        A->mult(x, b);
    }

    // Convert system to Hypre format 
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(&x);
    HYPRE_IJVector b_h_ij = convert(&b);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);

    for (int i = 0; i < num_tests; i++)
    {
        HYPRE_Solver solver_data;
        ParMultilevel* ml;

        x.set_const_value(0.0);
        double* x_h_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
        for (int i = 0; i < A->local_num_rows; i++)
        {
            x_h_data[i] = 0.0;
        }

        clear_cache(cache_array);

        // Create Hypre Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
                                coarsen_type, interp_type, p_max_elmts, agg_num_levels, 
                                strong_threshold);
        hypre_setup = MPI_Wtime() - t0;
        clear_cache(cache_array);

        // Solve Hypre Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        HYPRE_BoomerAMGSolve(solver_data, A_h, b_h, x_h);
        hypre_solve = MPI_Wtime() - t0;
        clear_cache(cache_array);

        // Delete hypre hierarchy
        hypre_BoomerAMGDestroy(solver_data);     

        // Setup Raptor Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);    
        t0 = MPI_Wtime();
        ml = new ParMultilevel(A, strong_threshold);
        raptor_setup = MPI_Wtime() - t0;
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

        // Solve Raptor Hierarchy
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        ml->solve(x, b);
        raptor_solve = MPI_Wtime() - t0;
        clear_cache(cache_array);

        MPI_Reduce(&hypre_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Hypre Setup Time: %e\n", t0);
        MPI_Reduce(&hypre_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Hypre Solve Time: %e\n", t0);

        MPI_Reduce(&raptor_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Raptor Setup Time: %e\n", t0);
        MPI_Reduce(&raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Raptor Solve Time: %e\n", t0);

        // Delete raptor hierarchy
        delete ml;
    }

    delete A;
    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);
    MPI_Finalize();

    return 0;
}

