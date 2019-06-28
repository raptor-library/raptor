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

TEST(ParPSRECGTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    bool print_res_tofile = true;
    bool check_soln = true;

    FILE* f;
    double val;

    //int grid[2] = {50, 50};
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParMultilevel *ml;
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals_t5;
    aligned_vector<double> residuals_t25;
    aligned_vector<double> residuals_t50;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    
    // Setup AMG hierarchy
    ml = new ParSmoothedAggregationSolver(0.0);
    ml->max_levels = 3;
    ml->setup(A);

    PSRECG(A, ml, x, b, 5, residuals_t5);

    if (rank == 0)
    {
        if (print_res_tofile)
        {
            f = fopen("psrecg_raptor_t5.txt", "w");
            for (int i = 0; i < residuals_t5.size(); i++)
            {
                fprintf(f, "%lg\n", residuals_t5[i]);
            }
            fclose(f);
        }
    }

    if (check_soln)
    {
        for (int i = 0; i < x.local_n; i++)
        {
            ASSERT_NEAR(x.local->values[i], 1.0, 1e-05);
        }        
    }

    x.set_const_value(0.0);
    SRECG(A, x, b, 25, residuals_t25);
    
    if (rank == 0)
    {
        if (print_res_tofile)
        {
            f = fopen("psrecg_raptor_t25.txt", "w");
            for (int i = 0; i < residuals_t25.size(); i++)
            {
                fprintf(f, "%lg\n", residuals_t25[i]);
            }
            fclose(f);
        }
    }
    
    if (check_soln)
    {
        for (int i = 0; i < x.local_n; i++)
        {
            ASSERT_NEAR(x.local->values[i], 1.0, 1e-05);
        }        
    }

    x.set_const_value(0.0);
    SRECG(A, x, b, 50, residuals_t50);
    
    if (rank == 0)
    {
        if (print_res_tofile)
        {
            f = fopen("psrecg_raptor_t50.txt", "w");
            for (int i = 0; i < residuals_t50.size(); i++)
            {
                fprintf(f, "%lg\n", residuals_t50[i]);
            }
            fclose(f);
        }
    }
    
    if (check_soln)
    {
        for (int i = 0; i < x.local_n; i++)
        {
            ASSERT_NEAR(x.local->values[i], 1.0, 1e-05);
        }        
    }

    delete[] stencil;
    delete ml;
    delete A;
    
} // end of TEST(ParPSRECGTest, TestsInKrylov) //



