// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_cg.hpp"
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

TEST(ParSRECGTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals_t5;
    aligned_vector<double> residuals_t25;
    aligned_vector<double> residuals_t50;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    double b_norm = b.norm(2);

    SRECG(A, x, b, 5, residuals_t5);

    if (rank == 0)
    {
        f = fopen("srecg_raptor_t5.txt", "w");
        for (int i = 0; i < residuals_t5.size(); i++)
        {
            fprintf(f, "%lg\n", residuals_t5[i]);
        }
        fclose(f);
    }

    //printf("%f %f %d\n", residuals_t5[0], residuals_t5[residuals_t5.size()-1], residuals_t5.size());
    
    /*FILE* f = fopen("../../../../test_data/srecg_t5_res.txt", "r");
    double val;
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &val);
        ASSERT_NEAR(residuals_t5[i], val, 1e-05);
    }
    fclose(f);*/

    x.set_const_value(0.0);
    SRECG(A, x, b, 25, residuals_t25);
    
    //printf("%f %f %d\n", residuals_t25[0], residuals_t25[residuals_t25.size()-1], residuals_t25.size());
    
    /*f = fopen("../../../../test_data/srecg_t25_res.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &val);
        ASSERT_NEAR(residuals_t25[i], val, 1e-06);
    }
    fclose(f);*/
    
    if (rank == 0)
    {
        f = fopen("srecg_raptor_t25.txt", "w");
        for (int i = 0; i < residuals_t25.size(); i++)
        {
            fprintf(f, "%lg\n", residuals_t25[i]);
        }
        fclose(f);
    }

    x.set_const_value(0.0);
    SRECG(A, x, b, 50, residuals_t50);
    
    //printf("%f %f %d\n", residuals_t50[0], residuals_t50[residuals_t50.size()-1], residuals_t50.size());
    
    /*f = fopen("../../../../test_data/srecg_t50_res.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &val);
        ASSERT_NEAR(residuals_t50[i], val, 1e-06);
    }
    fclose(f);*/
    
    if (rank == 0)
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
    
} // end of TEST(ParSRECGTest, TestsInKrylov) //



