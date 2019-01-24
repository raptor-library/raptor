// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_cg.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParSRECGTimings, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    bool print_res_tofile = false;

    int timings = 50;
    double start, stop, total, avg_time;

    FILE* f;
    double val;

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParMultilevel *ml;
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals;
    aligned_vector<double> pre_residuals;
    aligned_vector<double> residuals_t5;
    aligned_vector<double> residuals_t25;
    aligned_vector<double> residuals_t50;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    double b_norm = b.norm(2);
    
    /* --------- PCG Timings --------- */

    // Setup AMG hierarchy
    ml = new ParSmoothedAggregationSolver(0.0);
    ml->max_levels = 3;
    ml->setup(A);

    // AMG Preconditioned CG
    total = 0.0; 
    for (int i = 0; i < 50; i++)
    {
        x.set_const_value(0.0);
        start = MPI_Wtime();        
        PCG(A, ml, x, b, pre_residuals);
        stop = MPI_Wtime();
        total += (stop - start);
    }
    avg_time = total / timings;

    // Print timings to file
    //f = fopen("cg_raptor_timings.txt", "w");
    if (rank == 0) printf("- PCG -\n");
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%lg\n", avg_time);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //fclose(f);
    
    // Print res to file
    if (rank == 0 && print_res_tofile)
    {
        f = fopen("cg_raptor.txt", "w");
        for (int i = 0; i < residuals.size(); i++)
        {
            fprintf(f, "%lg\n", pre_residuals[i]);
        }
        fclose(f);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------- CG Timings ---------- */
   
    total = 0.0; 
    for (int i = 0; i < 50; i++)
    {
        x.set_const_value(0.0);
        start = MPI_Wtime();        
        CG(A, x, b, residuals);
        stop = MPI_Wtime();
        total += (stop - start);
    }
    avg_time = total / timings;

    // Print timings to file
    //f = fopen("cg_raptor_timings.txt", "w");
    if (rank == 0) printf("- CG -\n");
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%lg\n", avg_time);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //fclose(f);
    
    // Print res to file
    if (rank == 0 && print_res_tofile)
    {
        f = fopen("cg_raptor.txt", "w");
        for (int i = 0; i < residuals.size(); i++)
        {
            fprintf(f, "%lg\n", residuals[i]);
        }
        fclose(f);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* ---------- SRECG t = 5 Timings ---------- */

    total = 0.0; 
    for (int i = 0; i < 50; i++)
    {
        x.set_const_value(0.0);
        start = MPI_Wtime();        
        SRECG(A, x, b, 5, residuals_t5);
        stop = MPI_Wtime();
        total += (stop - start);
    }
    avg_time = total / timings;

    // Print timings to file
    //f = fopen("srecg_t5_raptor_timings.txt", "w");
    if (rank == 0) printf("- SRECG t5 -\n");
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%lg\n", avg_time);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //fclose(f);

    // Print res to file
    if (rank == 0 && print_res_tofile)
    {
        f = fopen("srecg_raptor_t5.txt", "w");
        for (int i = 0; i < residuals_t5.size(); i++)
        {
            fprintf(f, "%lg\n", residuals_t5[i]);
        }
        fclose(f);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* ---------- SRECG t = 25 Timings ---------- */

    total = 0.0; 
    for (int i = 0; i < 50; i++)
    {
        x.set_const_value(0.0);
        start = MPI_Wtime();        
        SRECG(A, x, b, 25, residuals_t25);
        stop = MPI_Wtime();
        total += (stop - start);
    }
    avg_time = total / timings;
    
    // Print timings to file
    //f = fopen("srecg_t25_raptor_timings.txt", "w");
    if (rank == 0) printf("- SRECG t25 -\n");
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%lg\n", avg_time);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //fclose(f);
    
    // Print res to file
    if (rank == 0 && print_res_tofile)
    {
        f = fopen("srecg_raptor_t25.txt", "w");
        for (int i = 0; i < residuals_t25.size(); i++)
        {
            fprintf(f, "%lg\n", residuals_t25[i]);
        }
        fclose(f);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    /* ---------- SRECG t = 50 Timings ---------- */

    total = 0.0; 
    for (int i = 0; i < 50; i++)
    {
        x.set_const_value(0.0);
        start = MPI_Wtime();        
        SRECG(A, x, b, 50, residuals_t50);
        stop = MPI_Wtime();
        total += (stop - start);
    }
    avg_time = total / timings;
    
    // Print timings to file
    //f = fopen("srecg_t50_raptor_timings.txt", "w");
    if (rank == 0) printf("- SRECG t50 -\n");
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%lg\n", avg_time);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //fclose(f);
    
    if (rank == 0 && print_res_tofile)
    {
        f = fopen("srecg_raptor_t50.txt", "w");
        for (int i = 0; i < residuals_t50.size(); i++)
        {
            fprintf(f, "%lg\n", residuals_t50[i]);
        }
        fclose(f);
    }

    delete[] stencil;
    delete A;
    delete ml;
    
} // end of TEST(ParSRECGTimings, TestsInKrylov) //
