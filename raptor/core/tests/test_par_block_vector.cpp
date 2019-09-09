// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParVectorTest, TestsInCore)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    int vecs_in_block = 4;
    int global_n = 100;
    int local_n = global_n / num_procs;
    int first_n = rank * ( global_n / num_procs);

    if (global_n % num_procs > rank)
    {
        local_n++;
        first_n += rank;
    }
    else
    {
        first_n += (global_n % num_procs);
    }

    BVector v(global_n, vecs_in_block);
    ParBVector v_par(global_n, local_n, vecs_in_block);

    v.set_const_value(1.0);
    v_par.set_const_value(1.0);

    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < local_n; i++)
        {
            ASSERT_EQ( v.values[k*global_n + i], v_par.local->values[k*local_n + i] );
        }
    }
    
    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < global_n; i++)
        {
            srand(k*global_n + i);
            v.values[k*global_n + i] = ((double)rand()) / RAND_MAX;
        }
    }
    
    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < local_n; i++)
        {
            srand(k*global_n + i);
            v_par.local->values[k*local_n + i] = ((double)rand()) / RAND_MAX;
        }
    }
    
    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < local_n; i++)
        {
            ASSERT_EQ( v.values[k*global_n + i], v_par.local->values[k*local_n + i] );
        }
    }

    // Test ParVector Append
    BVector p(global_n, vecs_in_block);
    ParBVector p_par(global_n, local_n, vecs_in_block);
    p.set_const_value(1.0);
    v.set_const_value(1.0);
    p_par.set_const_value(1.0);
    v_par.set_const_value(1.0);

    double* alphas = new double[vecs_in_block];
    for (int i = 0; i < vecs_in_block; i++) alphas[i] = i + 1.0;
    v.scale(1.0, alphas);
    v_par.scale(1.0, alphas);
    
    for (int i = 0; i < vecs_in_block; i++) alphas[i] += 1.0;
    p.scale(1.0, alphas);
    p_par.scale(1.0, alphas);

    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < global_n; i++)
        {
            v.values[k*global_n + i] = k + 1.0;
            p.values[k*global_n + i] = k + 2.0;
        }
    }
    
    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < local_n; i++)
        {
            v_par.local->values[k*local_n + i] = k + 1.0;
            p_par.local->values[k*local_n + i] = k + 2.0;
        }
    }

    v_par.append(p_par);
    v.append(p);

    ASSERT_EQ( v_par.global_n, v.num_values );
    ASSERT_EQ( v_par.local->b_vecs, v.b_vecs );
    
    for (int k = 0; k < vecs_in_block; k++)
    {
        for (int i = 0; i < local_n; i++)
        {
            ASSERT_EQ( v.values[k*global_n + i], v_par.local->values[k*local_n + i] );
        }
    }
    
} // end of TEST(ParBlockVectorTest, TestsInCore) //

