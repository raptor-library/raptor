#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "gallery/matrix_IO.hpp"


using namespace raptor;

int main(int argc, char* argv[])
{
    CSRMatrix* A = readMatrix("../../tests/rss_laplace_A0.mtx", 1);
    CSRMatrix* P = readMatrix("../../tests/rss_laplace_P0.mtx", 0);
    Vector x(P->n_cols);
    Vector b(P->n_rows);
    FILE* f;

    f = fopen("../../tests/rss_laplace_brand0.txt", "r");
    for (int i = 0; i < P->n_rows; i++)
    {
        fscanf(f, "%lg\n", &b[i]);
    }
    fclose(f);

    P->mult_T(b, x);
    f = fopen("../../tests/rss_laplace_x0.txt", "r");
    double val;
    for (int i = 0; i < P->n_cols; i++)
    {
        fscanf(f, "%lg\n", &val);
        assert(fabs(x[i] - val) < 1e-06);
    }
    fclose(f);


    delete A;
    delete P;

    return 0;
}
