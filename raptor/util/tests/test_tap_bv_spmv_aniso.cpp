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

TEST(TAPAnisoSpMVTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE *f;
    double b_val;
    int vecs_in_block = 3;
    int grid[2] = {25, 25};
    //int grid[2] = {4, 4};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    setenv("PPN", "4", 1);
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);

    for (int i = 0 ; i < A->on_proc->vals.size(); i++)
    {
        A->on_proc->vals[i] = 1.0;
    }

    for (int i = 0 ; i < A->off_proc->vals.size(); i++)
    {
        A->off_proc->vals[i] = 1.0;
    }

    ParBVector x(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, vecs_in_block);
    ParBVector b(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, vecs_in_block);
    
    ParVector x1(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    ParVector b1(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);

    ParVector x2(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    ParVector b2(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);

    x.set_const_value(1.0);
    x1.set_const_value(1.0);
    x2.set_const_value(2.0);
    
    A->tap_mult(x1, b1);
    A->tap_mult(x2, b2);

    for (int i=0; i<A->local_num_rows; i++)
    {
        x.local->values[A->local_num_rows + i] = 2.0;
    }

    A->tap_mult(x, b);
    
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
        //printf("b[%d] %lf b1[%d] %lf\n", i, b.local->values[i], i, b1.local->values[i]);
    }
    
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
        //printf("b[%d] %lf b2[%d] %lf\n", A->local_num_rows + i, b.local->values[i + A->local_num_rows], i, b2.local->values[i]);
    }

    /*printf("---\n");
    printf("---\n");
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%d ---\n", rank);
            b.local->print();
            printf("---\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);*/

    /*f = fopen("../../../../test_data/aniso_ones_b.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &b_val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    }
    fclose(f);*/

    // TEST SAME WITH SIMPLE TAP COMM
    /*delete A->tap_comm;
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);

    x.set_const_value(1.0);
    for (int i=0; i<A->on_proc_num_cols; i++)
    {
        x.local->values[A->on_proc_num_cols + i] = 2.0;
    }
    A->tap_mult(x, b);*/
    
    /*for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }*/

    /*for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("---\n");
            b.local->print();
            printf("---\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    printf("---\n");
    printf("---\n");
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < num_procs; i++)
    {
        if (i == rank)
        {
            printf("%d ---\n", rank);
            b2.local->print();
            printf("---\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);*/

    /*f = fopen("../../../../test_data/aniso_ones_b.txt", "r");
    for (int i = 0; i < A->partition->first_local_row; i++)
    {
        fscanf(f, "%lg\n", &b_val);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    }
    fclose(f);*/

    delete A;
    delete[] stencil;

    setenv("PPN", "16", 1);

} // end of TEST(ParAnisoSpMVTest, TestsInUtil) //
