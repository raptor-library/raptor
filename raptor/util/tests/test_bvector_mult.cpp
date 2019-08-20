// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(AnisoSpMVTest, TestsInUtil)
{
    int Q_bvecs = 5;
    int W_bvecs = 2;
    int len = 20;
    double val;

    BVector *Q = new BVector(len, Q_bvecs);
    BVector *W = new BVector(len, W_bvecs);
    BVector *B = new BVector(Q_bvecs, W_bvecs);

    Q->set_const_value(1.0);
    W->set_const_value(1.0);
    double* alphs = new double[W_bvecs];
    for(int i = 0; i < W_bvecs; i++)
    {
        alphs[i] = i + 1.0;
    }
    for (int i = 0; i < len; i++)
    {
        for (int v = 0; v < W_bvecs; v++)
        {
            W->values[v*len + i] = v + 1.0;
        }
    }
    //W->scale(1.0, *alphs);

    // Test B <- Q^T * W
    Q->mult_T(*W, *B);

    ASSERT_EQ(B->b_vecs, W_bvecs);
    ASSERT_EQ(B->num_values, Q_bvecs);

    int size = B->num_values;
    for (int i = 0; i < W_bvecs; i++)
    {
        val = len*(i+1);
        for (int j = 0; j < Q_bvecs; j++)
        {
            ASSERT_NEAR(B->values[i*size + j], val, 1e-06);
        }
    }

    // Test W <- Q * B
    W->set_const_value(0.0);
    Q->mult(*B, *W);
    
    ASSERT_EQ(W->b_vecs, B->b_vecs);
    ASSERT_EQ(W->num_values, len);
    
    size = W->num_values;
    for (int i = 0; i < B->b_vecs; i++)
    {
        val = Q_bvecs*len*(i+1);
        for (int j = 0; j < len; j++)
        {
            ASSERT_NEAR(W->values[i*size + j], val, 1e-06);
        }
    }

    delete Q;
    delete W;
    delete B;
    delete alphs;

} // end of TEST(AnisoSpMVTest, TestsInUtil) //

