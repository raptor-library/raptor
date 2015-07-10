#include <mpi.h>
#include <gtest/gtest.h>
#include <math.h>
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "util/linalg/spmv.hpp"


TEST(linag, spmv) 
{
//int main( int argc, char *argv[] )
//{
//    MPI_Init(&argc, &argv);

	data_t eps = 1.0;
	data_t theta = 0.0;

	index_t* grid = (index_t*) calloc(2, sizeof(index_t));
	grid[0] = 4;
	grid[1] = 4;

	index_t dim = 2;

	index_t rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	data_t* stencil = diffusion_stencil_2d(eps, theta);
	ParMatrix* A = stencil_grid(stencil, grid, dim, CSR);
    delete[] stencil;

	index_t global_num_rows = A->global_rows;
	index_t local_num_rows = A->local_rows;

	// Create the rhs and solution
	ParVector* b = new ParVector(global_num_rows, local_num_rows);
	ParVector* x = new ParVector(global_num_rows, local_num_rows);

	x->set_const_value(1.);
	b->set_const_value(0.);
	parallel_spmv(A, x, b, 1., 0.);

	for (index_t proc = 0; proc < num_procs; proc++)
	{
		if (proc == rank) {
			for (index_t i = 0; i < local_num_rows; i++)
			{
				data_t* data = (b->local)->data();
				printf("b[%d] = %2.3e\n", i+(A->first_col_diag), data[i]);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

    delete x;
    delete b;
    delete A;

//    MPI_Finalize();
}
