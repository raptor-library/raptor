#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "aggregation/seq/prolongation.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    int n_rows = 4;
    int n_cols = 4;
    double data[16] = {2.0, -1.0,  0.0,  0.0, 
        -1.0,  2.0, -1.0,  0.0,
         0.0, -1.0,  2.0, -1.0, 
         0.0,  0.0, -1.0,  2.0};

    CSRMatrix* A = new CSRMatrix(n_rows, n_cols, data);
    A->sort();
    A->print();

    CSRMatrix* S;
    CSRMatrix* T;
    CSRMatrix* P;

    S = A->strength(0.3);

    T = S->aggregate();
    printf("N_agg = %d\n", T->n_cols);
    T->print();

    P = jacobi_prolongation(A, T, 4.0/3, 2);
    P->sort();
    P->print();
}

