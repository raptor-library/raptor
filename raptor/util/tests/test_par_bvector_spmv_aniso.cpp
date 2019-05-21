// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParBVectorAnisoSpMVTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE* f;
    double b_val;
    int vecs_in_block = 3;
    //int grid[2] = {25, 25};
    int grid[2] = {5, 5};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    ParBVector *x = new ParBVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, vecs_in_block);
    ParBVector *b = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, vecs_in_block);

    x->set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0, 1.0};
    x->scale(1.0, &(alphas[0]));

    A->mult(*x, *b);

    /*f = fopen("../../../../test_data/aniso_ones_b.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &b_val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        for (int v = 0; v < vecs_in_block; v++)
        {
            if (v == 1) ASSERT_NEAR(b->local->values[i + v*b->local_n], 2*b_val, 1e-06);
            else ASSERT_NEAR(b->local->values[i + v*b->local_n], b_val, 1e-06);
        }
    }
    fclose(f);*/
    
    b->set_const_value(1.0);
    b->scale(1.0, &(alphas[0]));
    A->mult_T(*b, *x);
    /*f = fopen("../../../../test_data/aniso_ones_b_T.txt", "r");
    for (int i = 0; i < A->partition->first_local_col; i++)
    {
        fscanf(f, "%lg\n", &b_val);
    }
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        for (int v = 0; v < vecs_in_block; v++)
        {
            //ASSERT_NEAR(x->local->values[i + v*x->local_n], b_val, 1e-06);
            printf("%d v %d b_val %e x %e\n", rank, v, b_val, x->local->values[i + v*x->local_n]);
        }
    }
    fclose(f);*/

    /*for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        for (int v = 0; v < vecs_in_block; v++)
        {
            x->local->values[i + v*x->local_n] = A->partition->first_local_col + i;
        }
    }
    A->mult(*x, *b);
    f = fopen("../../../../test_data/aniso_inc_b.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &b_val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        for (int v = 0; v < vecs_in_block; v++)
        {
            ASSERT_NEAR(b->local->values[i + v*b->local_n], b_val, 1e-06);
        }
    }
    fclose(f);*/

    /*for (int i = 0; i < A->local_num_rows; i++)
    {
        b[i] = A->partition->first_local_row + i;
    }
    A->mult_T(b, x);
    f = fopen("../../../../test_data/aniso_inc_b_T.txt", "r");
    for (int i = 0; i < A->partition->first_local_col; i++)
    {
        fscanf(f, "%lg\n", &b_val);
    }
    for (int i = 0; i < A->on_proc_num_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(x[i], b_val, 1e-06);
    }
    fclose(f);*/

    delete x;
    delete b;
    delete A;
    delete[] stencil;

} // end of TEST(ParBVectorAnisoSpMVTest, TestsInUtil) //
