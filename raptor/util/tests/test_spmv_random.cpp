#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    double b_val;
    CSRMatrix* A = readMatrix("../../tests/random.mtx", 0);
    Vector x(A->n_cols);
    Vector b(A->n_rows);
    
    // Test b <- A*ones
    x.set_const_value(1.0);
    A->mult(x, b);
    FILE* f = fopen("../../tests/random_ones_b.txt", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(b[i] - b_val) < 1e-06);
    } 
    fclose(f);

    // Test b <- A_T*ones
    b.set_const_value(1.0);
    A->mult_T(b, x);
    f = fopen("../../tests/random_ones_b_T.txt", "r");
    for (int i = 0; i < A->n_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(x[i] - b_val) < 1e-06);
    } 
    fclose(f);

    // Tests b <- A*incr
    for (int i = 0; i < A->n_cols; i++)
    {
        x[i] = i;
    }
    A->mult(x, b);
    f = fopen("../../tests/random_inc_b.txt", "r");
    for (int i = 0; i < A->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(b[i] - b_val) < 1e-06);
    } 
    fclose(f);

    // Tests b <- A_T*incr
    for (int i = 0; i < A->n_rows; i++)
    {
        b[i] = i;
    }
    A->mult_T(b, x);
    f = fopen("../../tests/random_inc_b_T.txt", "r");
    for (int i = 0; i < A->n_cols; i++)
    {
        fscanf(f, "%lg\n", &b_val);
        assert(fabs(x[i] - b_val) < 1e-06);
    } 
    fclose(f);



}
