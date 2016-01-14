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

    index_t num_tests = 1;
    const char* mesh = "../../mfem/data/beam-tet.mesh";
    int order = 3;

    ParMatrix* A;
    ParVector* x;
    ParVector* b;

    mfem_laplace(&A, &x, &b, mesh, order);

    data_t t0, tfinal;

    t0 = MPI_Wtime();
    for (int i = 0; i < num_tests; i++)
    {
        parallel_spmv(A, x, b, 1.0, 0.0, 1);
    }
    tfinal = (MPI_Wtime() - t0) / num_tests;

    // Print Timings
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Time per SpMV: %2.3e\n", t0);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg Time per SpMV: %2.3e\n", t0 / num_procs);

    delete A;
    delete x;
    delete b;

    MPI_Finalize();

    return 0;
}

