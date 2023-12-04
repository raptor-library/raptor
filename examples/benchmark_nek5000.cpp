// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "raptor/core/types.hpp"
#include "raptor/multilevel/par_multilevel.hpp"
#include "raptor/external/hypre_wrapper.hpp"

using namespace raptor;

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Set HYPRE AMG Perferences
    int coarsen_type = 0; // CLJP = 0, Falgout = 6
    int interp_type = 0; // Classical 0, Direct 3 
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

    // Declare Variables
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);
    HYPRE_IJMatrix A_ij; 
    HYPRE_IJVector x_ij;
    HYPRE_IJVector b_ij;
    hypre_ParCSRMatrix* A_h;
    hypre_ParVector* x_h;
    hypre_ParVector* b_h;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;
    double t0;
    double hypre_setup, hypre_solve;
    double raptor_setup, raptor_solve, tap_raptor_solve;
    HYPRE_Solver solver_data;
    ParMultilevel* ml;

    const char* filename = "../../examples/N_16/A_fem";

    HYPRE_IJMatrixRead(filename, MPI_COMM_WORLD, HYPRE_PARCSR, &A_ij);
    HYPRE_IJMatrixGetObject(A_ij, (void**) &A_h);

 

    // Convert hypre matrix to raptor and copy vector values
    A = convert(A_h);
    x = ParVector(A->global_num_cols, A->on_proc_num_cols, 
            A->partition->first_local_col);
    b = ParVector(A->global_num_rows, A->local_num_rows, 
            A->partition->first_local_row);
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    // Get hypre vector values
    x_ij = convert(x);
    b_ij = convert(b);
    HYPRE_IJVectorGetObject(x_ij, (void**) &x_h);
    HYPRE_IJVectorGetObject(b_ij, (void**) &b_h);
    double* x_h_data = hypre_VectorData(hypre_ParVectorLocalVector(x_h));
    double* b_h_data = hypre_VectorData(hypre_ParVectorLocalVector(b_h));   

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

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParMultilevel(A, strong_threshold, CLJP, Classical, SOR,
            1, 1.0, 50, -1, 3);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);

    // Solve Raptor Hierarchy
    int iter;
    std::vector<double> res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    iter = ml->solve(x, b, res);
    raptor_solve = MPI_Wtime() - t0;
    clear_cache(cache_array);

    // TAP Solve Raptor Hierarchy
    x.set_const_value(0.0);
    std::vector<double> tap_res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    iter = ml->tap_solve(x, b, res, 3);
    tap_raptor_solve = MPI_Wtime() - t0;
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
    if (rank == 0) for (int i = 0; i <= iter; i++)
    {
        assert(fabs(res[i] - tap_res[i]) < 1e-06);
        printf("Res Norm[%d] = %e\n", i, res[i]);
    }

    MPI_Reduce(&hypre_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hypre Setup Time: %e\n", t0);
    MPI_Reduce(&hypre_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hypre Solve Time: %e\n", t0);

    MPI_Reduce(&raptor_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Setup Time: %e\n", t0);
    MPI_Reduce(&raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Solve Time: %e\n", t0);
    MPI_Reduce(&tap_raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("TAP Raptor Solve Time: %e\n", t0);

    delete ml;

    delete A;
    HYPRE_IJMatrixDestroy(A_ij);
    HYPRE_IJVectorDestroy(x_ij);
    HYPRE_IJVectorDestroy(b_ij);

    MPI_Finalize();
    return 0;
}

