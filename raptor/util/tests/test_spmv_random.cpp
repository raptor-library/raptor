// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.


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

TEST(RandomSpMVTest, TestsInUtil)
{
    double b_val;
    CSRMatrix* A = readMatrix((char *)"../../../../test_data/random.mtx", 0);
    Vector x(A->n_cols);
    Vector b(A->n_rows);
    
    // Test b <- A*ones
    x.set_const_value(1.0);
    A->mult(x, b);
    FILE* f = fopen("../../../../test_data/random_ones_b.txt", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    b.set_const_value(1.0);
    A->mult_T(b, x);
    f = fopen("../../../../test_data/random_ones_b_T.txt", "r");
    for (int i = 0; i < A->n_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(x[i],b_val, 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < A->n_cols; i++)
    {
        x[i] = i;
    }
    A->mult(x, b);
    f = fopen("../../../../test_data/random_inc_b.txt", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(b[i], b_val, 1e-06);
    } 
    fclose(f);
 
    // Tests b <- A_T*incr
    for (int i = 0; i < A->n_rows; i++)
    {
        b[i] = i;
    }
    A->mult_T(b, x);
    f = fopen("../../../../test_data/random_inc_b_T.txt", "r");
    for (int i = 0; i < A->n_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        ASSERT_NEAR(x[i], b_val, 1e-06);
    } 
    fclose(f);
    
} // end of TEST(RandomSpMVTest, TestsInUtil) //

