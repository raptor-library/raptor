#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"

using namespace raptor;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    char* mesh = argv[1];
    int num_tests = atoi(argv[2]);
    int num_elements = atoi(argv[3]);
    int order = atoi(argv[4]);
    int async = atoi(argv[5]);

    ParMatrix* A;
    ParVector* x;
    ParVector* b;

//    mfem_laplace(&A, &x, &b, mesh, num_elements, order);
    mfem_linear_elasticity(&A, &x, &b, mesh, num_elements, order);
//    mfem_electromagnetic_diffusion(&A, &x, &b, mesh, num_elements, order);

    long local_nnz = A->diag->nnz + A->offd->nnz;
    long global_nnz;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Nonzeros = %lu\n", global_nnz);

    data_t t0, tfinal;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < num_tests; i++)
    {
        parallel_spmv(A, x, b, 1.0, 0.0, async);
    }
    tfinal = (MPI_Wtime() - t0) / num_tests;

    // Print Timings
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Time per (ASYNC=%d) SpMV: %2.3e\n", async, t0);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg Time per (ASYNC=%d) SpMV: %2.3e\n", async, t0 / num_procs);

    delete A;
    delete x;
    delete b;

    MPI_Finalize();

    return 0;
}

