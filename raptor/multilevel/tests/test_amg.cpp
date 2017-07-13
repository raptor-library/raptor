#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "multilevel/seq/multilevel.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    double eps = 0.001;
    double theta = M_PI / 8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    int grid[2] = {25, 25};
    CSRMatrix* A = stencil_grid(stencil, grid, 2);
    A->sort();
    Vector x(A->n_rows);
    Vector b(A->n_rows);
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    data_t* B = new data_t[A->n_rows];
    for (int i = 0; i < A->n_rows; i++)
        B[i] = 1.0;

    Multilevel ml(A, B, 1, 0, 4.0/3, 1, 50);
    ml.solve(x, b);

    delete[] B;

    delete[] stencil;
}
