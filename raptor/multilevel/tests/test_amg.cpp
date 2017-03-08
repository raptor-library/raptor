#include <assert.h>

#include "core/types.hpp"
#include "multilevel/level.hpp"
#include "multilevel/multilevel.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/prolongation.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    CSRMatrix A;
    double eps = 0.001;
    double theta = M_PI / 8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    int grid[2] = {25, 25};
    stencil_grid(&A, stencil, grid, 2);
    A.sort();
    Vector x(A.n_rows);
    Vector b(A.n_rows);
    x.set_const_value(1.0);
    A.mult(x, b);
    x.set_rand_values();

    Multilevel ml(A, 0, 4.0/3, 1, 50);
    ml.solve(x, b);

    delete[] stencil;
}
