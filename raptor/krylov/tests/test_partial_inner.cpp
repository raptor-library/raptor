#include <assert.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/partial_inner.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm contig_comm, striped_comm;

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector y(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);

    x.set_const_value(1.0);
    y.set_const_value(1.0);

    data_t inner;
    int color, first_root, second_root, part_global; 

    // Contiguous Communicator
    create_partial_inner_comm(contig_comm, x, color, first_root, second_root, part_global, 1);

    // Contiguous Tests for both halves 
    inner = half_inner(contig_comm, x, y, color, 0, first_root, second_root, part_global);
    assert(fabs(inner - x.global_n) < 1e-01);

    MPI_Barrier(MPI_COMM_WORLD);

    inner = half_inner(contig_comm, x, y, color, 1, second_root, first_root, part_global);
    assert(fabs(inner - x.global_n) < 1e-01);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Striped Communicator
    create_partial_inner_comm(striped_comm, x, color, first_root, second_root, part_global, 0);
    
    // Striped Tests for both halves 
    inner = half_inner(striped_comm, x, y, color, 0, first_root, second_root, part_global);
    assert(fabs(inner - x.global_n) < 1e-01);

    MPI_Barrier(MPI_COMM_WORLD);

    inner = half_inner(striped_comm, x, y, color, 1, second_root, first_root, part_global);
    assert(fabs(inner - x.global_n) < 1e-01);
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm_free(&contig_comm);
    MPI_Comm_free(&striped_comm);
    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
