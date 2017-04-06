#include <assert.h>

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
#include "core/tap_comm.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

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

    ParCSRMatrix A;
    par_stencil_grid(&A, stencil, grid, 2);

    ParVector x(A.global_num_rows, A.local_num_rows, A.first_local_row);
    Vector& x_lcl = x.local;
    for (int i = 0; i < A.local_num_rows; i++)
    {
        x_lcl[i] = A.first_local_row + i;
    }

    TAPComm* tap_comm = new TAPComm(A.off_proc_column_map, A.first_local_row,
            A.first_local_col, A.global_num_cols, A.local_num_cols);

    Vector& tap_recv = x.communicate(tap_comm, MPI_COMM_WORLD);
    Vector& par_recv = x.communicate(A.comm, MPI_COMM_WORLD);
    assert(tap_recv.size == par_recv.size);
    for (int i = 0; i < par_recv.size; i++)
    {
        assert(fabs(par_recv[i] - tap_recv[i]) < zero_tol);
    }

    MPI_Finalize();
}




