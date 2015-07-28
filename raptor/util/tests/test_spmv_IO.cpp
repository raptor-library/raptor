#include <mpi.h>
#include <gtest/gtest.h>
#include <math.h>
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/matrix_IO.hpp"
#include "util/linalg/spmv.hpp"


TEST(linag, spmv) 
{
//int main( int argc, char *argv[] )
//{
//    MPI_Init(&argc, &argv);

	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char file[] = "LFAT5.mtx";
    //char file[] = "msc01440.mtx";
    //char file[] = "plbuckle.mtx";

	ParMatrix* A = readParMatrix(file, MPI_COMM_WORLD, true, 1);
    assert(A != NULL && "Error reading matrix!!!");
	int global_num_rows = A->global_rows;
	int local_num_rows = A->local_rows;

	// Create the rhs and solution
	ParVector* b = new ParVector(global_num_rows, local_num_rows);
	ParVector* x = new ParVector(global_num_rows, local_num_rows);

	x->set_const_value(1.);
	parallel_spmv(A, x, b, 1., 0.);

	for (int proc = 0; proc < num_procs; proc++)
	{
		if (proc == rank) {
			for (int i = 0; i < local_num_rows; i++)
			{
				double* data = (b->local)->data();
				printf("b[%d] = %2.3e\n", i+(A->first_col_diag), data[i]);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
//    MPI_Finalize();
}
