// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParBVectorMultTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double val;
    int Q_bvecs = 5;
    int W_bvecs = 2;
    //int global_n = 100;
    int global_n = 20;
    int local_n = global_n / num_procs;
    int first_n = rank * local_n;

    if (global_n % num_procs > rank)
    {
        local_n++;
        first_n += rank;
    }
    else
    {
        first_n += (global_n % num_procs);
    }

    BVector *Q = new BVector(global_n, Q_bvecs);
    BVector *W = new BVector(global_n, W_bvecs);
    BVector *B = new BVector(Q_bvecs, W_bvecs);

    ParBVector *Q_par = new ParBVector(global_n, local_n, Q_bvecs);
    ParBVector *W_par = new ParBVector(global_n, local_n, W_bvecs);
    BVector *B_par = new BVector(Q_bvecs, W_bvecs);

    // Setup BVector problem for comparison
    Q->set_rand_values();
    W->set_const_value(1.0);
    std::vector<double> alphas = {1.0, 2.0};
    W->scale(1.0, &(alphas[0]));
    /*for (int i = 0; i < Q_bvecs; i++)
    {
        for (int j = 0; j < global_n; j++)
        {
            Q->values[i*global_n + j] = (i+1) * (j+1);
        }
    }*/

    W->set_rand_values();
    Q->mult_T(*W, *B);

    // Test B_par <- Q_par^T * W_par
    Q_par->set_const_value(1.0);
    W_par->set_const_value(1.0);
    W_par->scale(1.0, &(alphas[0]));
    for (int i = 0; i < Q_bvecs; i++)
    {
        for (int j = 0; j < local_n; j++)
        {
            //Q_par->local->values[i*local_n + j] = (i+1) * (first_n + j + 1);
            Q_par->local->values[i*local_n + j] = Q->values[i*global_n + first_n + j];
        }
    }
    
    for (int i = 0; i < W_bvecs; i++)
    {
        for (int j = 0; j < local_n; j++)
        {
            //Q_par->local->values[i*local_n + j] = (i+1) * (first_n + j + 1);
            W_par->local->values[i*local_n + j] = W->values[i*global_n + first_n + j];
        }
    }

    Q_par->mult_T(*W_par, *B_par);

    /*for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            for (int i = 0; i < Q_bvecs; i++)
            {
                for (int j = 0; j < local_n; j++)
                {
                    printf("Q[%d] %e Q_par[%d] %e\n", i*local_n+j, Q_par->local->values[i*local_n+j], i*global_n+j, Q->values[i*global_n+j+first_n]);
                }
            }
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }*/

    ASSERT_EQ(B_par->b_vecs, B->b_vecs);
    ASSERT_EQ(B_par->num_values, B->num_values);

    for (int i = 0; i < B->num_values * B->b_vecs; i++)
    {
        //ASSERT_NEAR(B->values[i], B_par->values[i], 1e-06);
        printf("%d B[%d] %e Bpar[%d] %e\n", rank, i, B->values[i], i, B_par->values[i], 1e-06);
    }

    // Test W <- Q * B
    /*W->set_const_value(0.0);
    Q->mult(*B, *W);

    W_par->set_const_value(0.0);
    Q_par->mult(*B_par, *W_par);

    ASSERT_EQ(W_par->local->b_vecs, W->b_vecs);
    ASSERT_EQ(W_par->global_n, W->num_values);
    
    for (int k = 0; k < W->b_vecs; k++)
    {
        for (int i = 0; i < local_n; i++)
        {
            ASSERT_NEAR(W->values[k*global_n+i], W_par->local->values[k*local_n+i], 1e-06);
        } 
    }*/

    delete Q;
    delete W;
    delete B;
    delete Q_par;
    delete W_par;
    delete B_par;

} // end of TEST(ParBVectorMultTest, TestsInUtil) //
