// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(RandomBSRSpMVTest, TestsInUtil)
{
/*    const char* rand_fn = "../../../../test_data/random.pm";
    const char* b_ones = "../../../../test_data/random_ones_b.txt";
    const char* b_T_ones = "../../../../test_data/random_ones_b_T.txt";
    const char* b_inc = "../../../../test_data/random_inc_b.txt";
    const char* b_T_inc = "../../../../test_data/random_inc_b_T.txt";

    double b_val;
    const CSRMatrix* A = readMatrix(rand_fn);
    BSRMatrix* B = new BSRMatrix(A, 2, 2);

    Vector x(A->n_cols);
    Vector b(A->n_rows);
    
    // Test b <- A*ones
    x.set_const_value(1.0);
    B->mult(x, b);
    FILE* f = fopen(b_ones, "r");
    for (int i = 0; i < B->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    b.set_const_value(1.0);
    B->mult_T(b, x);
    f = fopen(b_T_ones, "r");
    for (int i = 0; i < B->n_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(x[i],b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < B->n_cols; i++)
    {
        x[i] = i;
    }
    B->mult(x, b);
    f = fopen(b_inc, "r");
    for (int i = 0; i < B->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);
 
    // Tests b <- A_T*incr
    for (int i = 0; i < B->n_rows; i++)
    {
        b[i] = i;
    }
    B->mult_T(b, x);
    f = fopen(b_T_inc, "r");
    for (int i = 0; i < B->n_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(x[i], b_val, 1e-06);
    } 
    fclose(f);
    
    delete A;
    delete B;*/
} // end of TEST(RandomSpMVTest, TestsInUtil) //

