#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

using namespace raptor;

void compare(Vector& b, ParVector& b_par)
{
    double b_norm = b.norm(2);
    double b_par_norm = b_par.norm(2);

    assert(fabs(b_norm - b_par_norm) < 1e-06);

    Vector& b_par_lcl = b_par.local;
    for (int i = 0; i < b_par.local_n; i++)
    {
        assert(fabs(b_par_lcl[i] - b[i+b_par.first_local]) < 1e-06);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI/8.0;
    int grid[2] = {10, 10};
    int dense_n = 100;
    double* stencil = diffusion_stencil_2d(eps, theta);

    // Create Sequential Matrix A on each process
    CSRMatrix A;
    stencil_grid(&A, stencil, grid, 2);
    Vector x(A.n_cols);
    Vector b(A.n_rows);
    x.set_const_value(1.0);

    // Create Parallel Matrix A_par (and vectors x_par
    // and b_par) and mult b_par <- A_par*x_par
    ParCSRMatrix A_par;
    par_stencil_grid(&A_par, stencil, grid, 2);
    ParVector x_par(A_par.global_num_cols, A_par.local_num_cols, A_par.first_local_col);
    ParVector b_par(A_par.global_num_rows, A_par.local_num_rows, A_par.first_local_row);
    x_par.set_const_value(1.0);

    A.mult(x, b);
    A_par.mult(x_par, b_par);
    compare(b, b_par);

    // Set x and x_par to same random values
    for (int i = 0; i < x.size; i++)
    {
        srand(i);
        x[i] = ((double)rand()) / RAND_MAX;
    }
    for (int i = 0; i < x_par.local_n; i++)
    {
        srand(i+x_par.first_local);
        x_par.local[i] = ((double)rand()) / RAND_MAX;
    }
    A.mult(x, b);
    A_par.mult(x_par, b_par);
    compare(b, b_par);

    delete[] stencil;

    MPI_Finalize();
}

