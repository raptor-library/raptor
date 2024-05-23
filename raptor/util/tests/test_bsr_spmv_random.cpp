// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"
#include "raptor/raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

} // end of main() //

TEST(RandomSpMVTest, TestsInUtil)
{
    const char* rand_fn = "../../../../test_data/random.pm";
    const char* b_ones = "../../../../test_data/random_ones_b.txt";
    const char* b_T_ones = "../../../../test_data/random_ones_b_T.txt";
    const char* b_inc = "../../../../test_data/random_inc_b.txt";
    const char* b_T_inc = "../../../../test_data/random_inc_b_T.txt";

    double b_val;
    CSRMatrix* A = readMatrix(rand_fn);
    Vector x(A->n_cols);
    Vector b(A->n_rows);
    
    BSRMatrix* A_bsr = new BSRMatrix(A, 5, 5);

    // Test b <- A*ones
    int n_items_read;
    x.set_const_value(1.0);
    A_bsr->mult(x, b);
    FILE* f = fopen(b_ones, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    b.set_const_value(1.0);
    A_bsr->mult_T(b, x);
    f = fopen(b_T_ones, "r");
    for (int i = 0; i < A->n_cols; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(x[i],b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < A->n_cols; i++)
    {
        x[i] = i;
    }
    A_bsr->mult(x, b);
    f = fopen(b_inc, "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);
 
    // Tests b <- A_T*incr
    for (int i = 0; i < A->n_rows; i++)
    {
        b[i] = i;
    }
    A_bsr->mult_T(b, x);
    f = fopen(b_T_inc, "r");
    for (int i = 0; i < A->n_cols; i++)
    {
        n_items_read = fscanf(f, "%lg\n", &b_val);
        ASSERT_EQ(n_items_read, 1);
        ASSERT_NEAR(x[i], b_val, 1e-06);
    } 
    fclose(f);

    delete A_bsr;
    delete A;
    
} // end of TEST(RandomSpMVTest, TestsInUtil) //

