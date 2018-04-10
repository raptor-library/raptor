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
    MPI_Comm first_comm, second_comm;

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector y(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);

    x.set_const_value(1.0);
    y.set_const_value(1.0);

    data_t inner;

    // Tests Setup
    int color=0, half_procs, part_global1=0, part_global2=0, part_global;

    if (num_procs > 1){
        half_procs = num_procs/2;
        if (num_procs % 2 != 0) half_procs++;
        if (rank >= half_procs) color++;

        //MPI_Comm first_comm, second_comm;
        int first_comm_size, second_comm_size;
        if (!color){
            MPI_Comm_split(MPI_COMM_WORLD, color, rank, &first_comm);
	    MPI_Comm_size(first_comm, &first_comm_size);
	    part_global1 = x.local_n;
            if (first_comm_size > 1) MPI_Allreduce(MPI_IN_PLACE, &part_global1, 1, MPI_INT, MPI_SUM, first_comm);
        }
        else{
 	    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &second_comm);
	    MPI_Comm_size(second_comm, &second_comm_size);
        }
    
        if (rank == 0) MPI_Send(&part_global1, 1, MPI_INT, half_procs, 1, MPI_COMM_WORLD);
        if (rank == half_procs) MPI_Recv(&part_global1, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (color && 1 < second_comm_size) MPI_Bcast(&part_global1, 1, MPI_INT, 0, second_comm);

        //if (!color) MPI_Comm_free(&first_comm);
        //else MPI_Comm_free(&second_comm);

        part_global2 = x.global_n - part_global1;
    }
    else{
	part_global1 = x.global_n;
	part_global2 = x.global_n;
    }

    // Test both inner product halves
    inner = half_inner_contig(x, y, 0, part_global1);
    assert(fabs(inner - x.global_n) < 1e-06);

    MPI_Barrier(MPI_COMM_WORLD);

    inner = half_inner_contig(x, y, 1, part_global2);
    assert(fabs(inner - x.global_n) < 1e-06);

    if (num_procs > 1){
        if (!color) MPI_Comm_free(&first_comm);
        else MPI_Comm_free(&second_comm);
    }

    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0;
}
