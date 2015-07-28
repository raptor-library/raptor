#include <mpi.h>
#include <math.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/diagonal.hpp"
#include "util/linalg/spmv.hpp"
using namespace raptor;
int main(int argc, char *argv[])
{
	int rank, num_procs;
    
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // matrix to read
    //char file[] = "LFAT5.mtx";
    //char file[] = "msc01440.mtx";
    //char file[] = "plbuckle.mtx";
    char file[] = "bcsstm25.mtx";

	// Create the matrix, rhs, and solution
	ParMatrix* A = readParMatrix(file, MPI_COMM_WORLD, true, 0);
    //ParMatrix* A = diagonal(100); 
    assert(A != NULL && "Error reading matrix!!!");
	int global_num_rows = A->global_rows;
	int local_num_rows = A->local_rows;

	ParVector* b = new ParVector(global_num_rows, local_num_rows);
	ParVector* x = new ParVector(global_num_rows, local_num_rows);
	x->set_const_value(1.0);

    // Time the SpMV
	double t0 = MPI_Wtime();
    parallel_spmv(A, x, b, 1., 0.);
    double total_time = MPI_Wtime() - t0;
    MPI_Reduce(&total_time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Max time for SpMV: %g\n", t0);
    MPI_Reduce(&total_time, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Min time for SpMV: %g\n", t0);
    MPI_Reduce(&total_time, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Avg time for SpMV: %g\n", t0/num_procs);
	

    parallel_spmv(A, x, b, -1.0, 1.0);

    double norm = b->norm<2>();
    if (rank == 0) printf("2Norm = %2.3e\n", norm);

    // Display result vector to verify correctness
    //for (int proc = 0; proc < num_procs; proc++)
	//{
	//	if (proc == rank) {
	//		for (int i = 0; i < local_num_rows; i++)
	//		{
	//			double* data = (b->local)->data();
	//			printf("b[%d] = %2.3e\n", i+(A->first_col_diag), data[i]);
	//		}
	//	}
	//	MPI_Barrier(MPI_COMM_WORLD);
	//}

    MPI_Finalize();

    return 0;
}
