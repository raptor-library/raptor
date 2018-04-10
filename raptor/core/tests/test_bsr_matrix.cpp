// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
using namespace raptor;


int main(int argc, char** argv)
{
    std::vector<int> row_ptr = {0,2,3,5};
    std::vector<int> cols = {0,1,1,1,2};
    std::vector<double> vals = {1.0, 0.0, 2.0, 1.0, 6.0, 7.0, 8.0, 2.0, 1.0, 4.0, 5.0, 1.0,
                                4.0, 3.0, 0.0, 0.0, 7.0, 2.0, 0.0, 0.0};


    int rows_in_block = 2;
    int cols_in_block = 2;
    int n = 6;

    // Create BSR Matrix (6x6)
    BSRMatrix A_bsr(n, n, rows_in_block, cols_in_block, row_ptr, cols, vals);

    Vector x(6);
    Vector b(6);
    Vector r(6);
    x.set_const_value(1.0);
    b.set_const_value(0.0);
    r.set_const_value(0.0);
   
    A_bsr.print();

    printf("\n-----------------------\n");
    printf("A * x = b\n");
    printf("-----------------------\n");
    A_bsr.mult(x, b);
    b.print();

    b.set_const_value(0.0);
    A_bsr.mult_T(x, b);
    printf("\n-----------------------\n");
    printf("A^T * x = b\n");
    printf("-----------------------\n");
    b.print();

    b.set_const_value(2.0);
    A_bsr.residual(x, b, r);
    printf("\n-----------------------\n");
    printf("r = b - A * x\n");
    printf("-----------------------\n");
    r.print();

    return 0;

    //::testing::InitGoogleTest(&argc, argv);
    //return RUN_ALL_TESTS();

} // end of main() //

TEST(MatrixTest, TestsInCore)
{
    std::vector<int> row_ptr = {0,2,3,5};
    std::vector<int> cols = {0,1,1,1,2};
    std::vector<double> vals = {1.0, 0.0, 2.0, 1.0, 6.0, 7.0, 8.0, 2.0, 1.0, 4.0, 5.0, 1.0,
                                4.0, 3.0, 0.0, 0.0, 7.0, 2.0, 0.0, 0.0};


    int rows_in_block = 2;
    int cols_in_block = 2;
    int n = 6;

    // Create BSR Matrix (6x6)
    BSRMatrix A_bsr(n, n, rows_in_block, cols_in_block, row_ptr, cols, vals);

    Vector x(6);
    Vector b(6);
    Vector r(6);
    x.set_const_value(1.0);
    b.set_const_value(0.0);
    r.set_const_value(0.0);

    A_bsr.mult(x, b);

    b.print();

    /*// -------------------------- UPDATE BELOW THIS

    // Add Values to COO Matrix
    A_coo.add_value(22, 5, 2.0);
    A_coo.add_value(17, 18, 1.0);
    A_coo.add_value(12, 21, 0.5);
    A_coo.add_value(0, 0, 1.0);
    A_coo.add_value(5, 7, 2.0);
    A_coo.add_value(7, 7, 1.0);
    A_coo.add_value(1, 0, 1.2);
    A_coo.add_value(0, 1, 2.2);
    A_coo.add_value(0, 0, 1.5);
    A_coo.add_value(12, 21, -1.0);

    // Check dimensions of A_coo
    ASSERT_EQ(A_coo.n_rows, 25);
    ASSERT_EQ(A_coo.n_cols, 25);
    ASSERT_EQ(A_coo.nnz, 10);

   // Check that rows, columns, and values in A_coo are correct
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(A_coo.idx1[i], rows[i]);
        ASSERT_EQ(A_coo.idx2[i], cols[i]);
        ASSERT_EQ(A_coo.vals[i], vals[i]);
    }

    // Create CSR Matrix from COO
    CSRMatrix A_csr(&A_coo);

    // Check dimensions of A_csr
    ASSERT_EQ(A_csr.n_rows,25);
    ASSERT_EQ(A_csr.n_cols,25);
    ASSERT_EQ(A_csr.nnz,10);

    // Check that rows, columns, and values in A_coo are correct
    
    for (int i = 0; i < 26; i++)
    {
        ASSERT_EQ(A_csr.idx1[i],row_ctr[i]);
    }*/

} // end of TEST(MatrixTest, TestsInCore) //

