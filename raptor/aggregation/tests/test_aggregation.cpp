#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/prolongation.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    int n_rows = 4;
    int n_cols = 4;
    double data[16] = {2.0, -1.0,  0.0,  0.0, 
        -1.0,  2.0, -1.0,  0.0,
         0.0, -1.0,  2.0, -1.0, 
         0.0,  0.0, -1.0,  2.0};

    CSRMatrix A(n_rows, n_cols, data);
    A.sort();
    A.print();

    CSRMatrix S(n_rows, n_cols, 0.5*A.nnz);
    A.symmetric_strength(&S, 0.3);

    CSCMatrix T;
    std::vector<int> c_points;

    int n_agg = standard_aggregation(S, T, c_points);
    printf("N_agg = %d\n", n_agg);
    T.print();

    CSCMatrix P;
    jacobi_prolongation(A, T, P, 4.0/3, 2);
    P.sort();
    P.print();
}

