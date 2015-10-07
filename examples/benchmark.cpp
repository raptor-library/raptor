#include <mpi.h>
#include <math.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"

#include "util/linalg/spmv.hpp"
//using namespace raptor;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    index_t n = 10;

    index_t num_tests = 1;

//    data_t eps = 1.0;
//    data_t theta = 0.0;
//    index_t* grid = (index_t*) calloc(2, sizeof(index_t));
//    grid[0] = n;
//    grid[1] = n;
//    index_t dim = 2;
//    data_t* stencil = diffusion_stencil_2d(eps, theta);
//    ParMatrix* A = stencil_grid(stencil, grid, dim, CSR);
//    delete[] stencil;
    
    // Create the matrix, rhs, and solution
    //char file[] = "LFAT5.mtx";
    char file[] = "msc01440.mtx";
    //char file[] = "plbuckle.mtx";
    //char file[] = "bcsstm25.mtx";
    ParMatrix* A = readParMatrix(file, MPI_COMM_WORLD, true, 1);

    int global_num_rows = A->global_rows;
    int local_num_rows = A->local_rows;
    ParVector* b = new ParVector(global_num_rows, local_num_rows);
    ParVector* x = new ParVector(global_num_rows, local_num_rows);
    x->set_const_value(1.0);

    data_t t0 = 0.0;
    data_t total_time = 0.0;

    t0 = MPI_Wtime();
    for (index_t i = 0; i < num_tests; i++)
    {
        parallel_spmv(A, x, b, 1., 0., 1);
    }
    total_time = (MPI_Wtime() - t0) / num_tests;
    MPI_Reduce(&total_time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Max time for Parallel SpMV: %g\n", t0);
    MPI_Reduce(&total_time, &t0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Min time for Parallel SpMV: %g\n", t0);
    MPI_Reduce(&total_time, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Avg time for Parallel SpMV: %g\n", t0/num_procs);
    
    double norm = b->norm(2);
    if (rank == 0) printf("2Norm = %2.3e\n", norm);

    MPI_Finalize();

    return 0;
}

