// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/vector.hpp"
#include "core/par_vector.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParBlockVectorTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
    ParBVector v_par(global_n, local_n, first_n, vecs_in_block);
    
    v.set_const_value(1.0);
    v_par.set_const_value(1.0);

    for (int i = 0; i < local_n*vecs_in_block; i++)
    {
        ASSERT_EQ( v.values[first_n*vecs_in_block+i], v_par.local->values[i] );
    }
    for (int i = 0; i < global_n*vecs_in_block; i++)
    {
        srand(i);
        v.values[i] = ((double)rand()) / RAND_MAX;
    }
    for (int i = 0; i < local_n*vecs_in_block; i++)
    {
        srand(i+first_n*vecs_in_block);
        v_par.local->values[i] = ((double)rand()) / RAND_MAX;
    }

    for (int i = 0; i < local_n*vecs_in_block; i++)
    {
        ASSERT_EQ(v.values[first_n*vecs_in_block+i], v_par.local->values[i]);
    }
    
    v_par.set_const_value(1.0);
    v.set_const_value(1.0);
    
    // Test ParBVector AXPY with ParBVector
    v.axpy(v, 0.5);
    v_par.axpy(v_par, 0.5);
    
    for (int i = 0; i < local_n*vecs_in_block; i++)
    {
        ASSERT_EQ( v[first_n*vecs_in_block+i], v_par.local->values[i] );
    }

    // Test ParBVector Norm
    double *norms = new double[v.b_vecs];
    double *norms_par = new double[v_par.local->b_vecs];
    double temp = v.norm(2, norms); 
    temp = v_par.norm(2, norms_par);
    for (int i = 0; i < vecs_in_block; i++)
    {
        ASSERT_EQ( norms[i], norms_par[i] );
    }

    ParVector c_par(global_n, local_n, first_n);
    c_par.set_const_value(1.0);

    Vector c(global_n);
    c.set_const_value(1.0);

    // Test ParBVector Mult_T with ParVector
    double *b = new double[v.b_vecs];
    double *inner_prods = new double[v.b_vecs];
    v_par.mult_T(c_par, b);
    temp = v.inner_product(c, inner_prods);
    for (int i = 0; i < vecs_in_block; i++)
    {
        ASSERT_EQ( inner_prods[i], b[i] );
    }

    // Test ParBVector Mult with Vector
    v_par.set_const_value(1.0);
    v.set_const_value(1.0);
    v_par.mult(c, c_par);
    temp = v_par.local->values[0]*vecs_in_block;
    for (int i = 0; i < local_n; i++)
    {
        ASSERT_EQ( temp, c_par.local->values[i] );
    }

    delete inner_prods;
    delete norms;
    delete norms_par;
    delete b;

} // end of TEST(ParVectorTest, TestsInCore) //

