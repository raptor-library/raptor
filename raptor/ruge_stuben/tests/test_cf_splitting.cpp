#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"
#include "ruge_stuben/cf_splitting.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {10, 10};
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    A->on_proc->sort();
    A->off_proc->sort();
    ParCSRMatrix* S = A->strength(0.0);

    std::vector<int> states;
    cf_splitting(S, states);
    for (int i = 0; i < S->local_num_rows; i++)
    {
        printf("States[%d] = %d\n", i + S->first_local_row, states[i]);
    }

    delete S;
    delete A;

    delete[] stencil;

    MPI_Finalize();
    return 0;
}


