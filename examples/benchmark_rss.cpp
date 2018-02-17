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
#include "tests/hypre_compare.hpp"
#include "gallery/external/hypre_wrapper.hpp"

#ifdef USING_MFEM
//#include "gallery/external/mfem_wrapper.hpp"
#endif

//using namespace raptor;
//

void form_hypre_weights(std::vector<double>& weights, int n_rows)
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
    ParVector x;
    ParVector b;

    double t0;
    double hypre_setup, hypre_solve;
    double raptor_setup, raptor_solve;
    double raptor_tap_solve;

    //int coarsen_type = 8; // PMIS
    int coarsen_type = 0; // CLJP
    //int coarsen_type = 6; // FALGOUT
    //int interp_type = 3; // Direct Interp
    //int interp_type = 0; // Classical Mod Interp
    int interp_type = 6;
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

    int cache_len = 10000;

    std::vector<double> cache_array(cache_len);
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
    /*else if (system == 2)
    {
        const char* mesh_file = argv[2];
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
    }*/
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

    // Convert system to Hypre format 
/*    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(x);
    HYPRE_IJVector b_h_ij = convert(b);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);

    HYPRE_Solver solver_data;

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
    clear_cache(cache_array);
*/
    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParMultilevel(A, strong_threshold, HMIS, Extended, SOR,
            1, 1.0, 50, -1, -1);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);

    ParCSRMatrix* Al;
    ParCSRMatrix* Pl;
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        Al = ml->levels[i]->A;
        Pl = ml->levels[i]->P;

        if (!Al->tap_comm)
        {
            Al->tap_comm = new TAPComm(Al->partition, Al->off_proc_column_map,
                    Al->on_proc_column_map);
        }

        if (!Pl->tap_comm)
        {
            Pl->tap_comm = new TAPComm(Pl->partition, Pl->off_proc_column_map,
                    Pl->on_proc_column_map);
        }
    }
    Al = ml->levels[ml->num_levels-1]->A;
    if (!Al->tap_comm)
    {
        Al->tap_comm = new TAPComm(Al->partition, Al->off_proc_column_map,
                Al->on_proc_column_map);
    }

    // Solve Raptor Hierarchy
    x.set_const_value(0.0);
    std::vector<double> res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml->solve(x, b, res);
    raptor_solve = MPI_Wtime() - t0;
    clear_cache(cache_array);

    // TAP Solve Raptor
    /*x.set_const_value(0.0);
    std::vector<double> tap_res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml->tap_solve(x, b, tap_res, 0);
    raptor_tap_solve = MPI_Wtime() - t0;
    clear_cache(cache_array);
*/

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
            printf("Res[%d] = %e\n", i, res[i]);
        }
    }   

    MPI_Reduce(&hypre_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hypre Setup Time: %e\n", t0);
    MPI_Reduce(&hypre_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hypre Solve Time: %e\n", t0);

    MPI_Reduce(&raptor_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Setup Time: %e\n", t0);
    MPI_Reduce(&raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Solve Time: %e\n", t0);
    MPI_Reduce(&raptor_tap_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor TAP Solve Time: %e\n", t0);
      
    int n0, s0;
    MPI_Reduce(&ml->setup_comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ml->setup_comm_n, &n0, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ml->setup_comm_s, &s0, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Setup Comm Time: %e, Comm N: %d, Comm S: %d\n", t0, n0, s0);
    MPI_Reduce(&ml->solve_comm_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ml->solve_comm_n, &n0, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ml->solve_comm_s, &s0, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Solve Comm Time: %e, Comm N: %d, Comm S: %d\n", t0, n0, s0);

    // Delete raptor hierarchy
//    delete ml;

//    HYPRE_IJMatrixDestroy(A_h_ij);
//    HYPRE_IJVectorDestroy(x_h_ij);
//    HYPRE_IJVectorDestroy(b_h_ij);
    delete A;
    MPI_Finalize();

    return 0;
}

