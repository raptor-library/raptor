#include <mpi.h>
#include <math.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/stencil.hpp"
#include "gallery/diffusion.hpp"
#include "util/linalg/spmv.hpp"

#include <assert.h>

//using namespace raptor;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    assert(num_procs > 1 && "num_procs == 1\n");

    index_t n = 50;

    index_t num_tests = 1;

    data_t eps = 0.1;
    data_t theta = 0.0;
    index_t* grid = new index_t[2];
    grid[0] = n;
    grid[1] = n;
    index_t dim = 2;
    data_t* stencil = diffusion_stencil_2d(eps, theta);
    ParMatrix* A = stencil_grid(stencil, grid, dim, CSR);
    delete[] stencil;
    delete[] grid;
    
    int global_num_rows = A->global_rows;
    int local_num_rows = A->local_rows;
    int global_num_cols = A->global_cols;
    int local_num_cols = A->local_cols;
    int first_row = A->first_row;
    int first_col_diag = A->first_col_diag;

    ParVector* b = new ParVector(global_num_cols, local_num_cols, first_col_diag);
    ParVector* x = new ParVector(global_num_rows, local_num_rows, first_row);
    ParVector* result = new ParVector(global_num_cols, local_num_cols, first_col_diag);

    b->set_const_value(0.0);
    x->set_const_value(1.0);
    result->set_const_value(0.0);

    data_t t0, tfinal;

    t0 = MPI_Wtime();
    for (int i = 0; i < num_tests; i++)
    {
        parallel_spmv(A, x, b, 1.0, 0.0, 0);
    }
    tfinal = (MPI_Wtime() - t0) / num_tests;

    // Print Timings
    long level_nnz, level_nnz_local;
    level_nnz_local = A->diag->nnz + A->offd->nnz;
    MPI_Reduce(&level_nnz_local, &level_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("A has %d nonzeros\n", level_nnz);
    double b_norm = b->norm(2);
    if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);

    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Time per SpMV: %2.3e\n", t0);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg Time per SpMV: %2.3e\n", t0 / num_procs);

    delete A;
    delete x;
    delete b;
    delete result;

    MPI_Finalize();

    return 0;
}

