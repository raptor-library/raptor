#include <assert.h>

#include "core/types.hpp"
#include "core/seq/matrix.hpp"
#include "core/seq/vector.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    int rows[10] = {22, 17, 12, 0, 5, 7, 1, 0, 0, 12};
    int cols[10] = {5, 18, 21, 0, 7, 7, 0, 1, 0, 21};
    double vals[10] = {2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 1.2, 2.2, 1.5, -1.0};

    int row_ctr[26] = {0, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 9, 
        9, 9, 9, 9, 10, 10, 10};
    int col_ctr[26] = {0, 3, 4, 4, 4, 4, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 10, 10, 10, 10};

    int row_ctr_condensed[8] = {0, 3, 4, 5, 6, 8, 9, 10};
    int col_ctr_condensed[7] = {0, 3, 4, 5, 7, 8, 10};

    // Create COO Matrix (25x25)
    COOMatrix A_coo(25, 25, 1);

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
    assert(A_coo.n_rows == 25);
    assert(A_coo.n_cols == 25);
    assert(A_coo.nnz == 10);

    // Check that rows, columns, and values in A_coo are correct
    for (int i = 0; i < 10; i++)
    {
        assert(A_coo.idx1[i] == rows[i]);
        assert(A_coo.idx2[i] == cols[i]);
        assert(A_coo.vals[i] == vals[i]);
    }

    // Create CSR Matrix from COO
    CSRMatrix A_csr(&A_coo);

    // Check dimensions of A_csr
    assert(A_csr.n_rows == 25);
    assert(A_csr.n_cols == 25);
    assert(A_csr.nnz == 10);

    // Check that rows, columns, and values in A_coo are correct
    int ctr = 0;
    for (int i = 0; i < 26; i++)
    {
        printf("Acsr idx[%d] = %d, row_ctr[%d] = %d\n", i, A_csr.idx1[i], i, row_ctr[i]);
        assert(A_csr.idx1[i] == row_ctr[i]);
    }



}
