#include <assert.h>
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "util/linalg/external/repartition.hpp"

using namespace raptor;

void compare(Vector& b, ParVector& b_par)
{
    double b_norm = b.norm(2);
    double b_par_norm = b_par.norm(2);

    assert(fabs(b_norm - b_par_norm) < 1e-06);

    Vector& b_par_lcl = b_par.local;
    for (int i = 0; i < b_par.local_n; i++)
    {
        assert(fabs(b_par_lcl[i] - b[i+b_par.first_local]) < 1e-06);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;

    // Create Sequential Matrix A on each process
    char* filename = "../../../../examples/LFAT5.mtx";
    if (argc > 1) filename = argv[1];

    ParCSRMatrix* A_orig = readParMatrix(filename, MPI_COMM_WORLD, 
            true, 1);
    A_orig->tap_comm = new TAPComm(A_orig->partition, A_orig->off_proc_column_map);
    ParVector x_orig(A_orig->global_num_rows, A_orig->local_num_rows, 
            A_orig->partition->first_local_row);
    ParVector b_orig(A_orig->global_num_rows, A_orig->local_num_rows, 
            A_orig->partition->first_local_row);

    int n_tests = 100;

    // TIME Original SpMV
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        A_orig->mult(x_orig, b_orig);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Orig SpMV Time %e\n", t0);

    // Time TAPSpMV
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        A_orig->tap_mult(x_orig, b_orig);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Orig TAPSpMV Time %e\n", t0);

    // Time Graph Partitioning
    t0 = MPI_Wtime();
    int* proc_part = ptscotch_partition(A_orig);
    for (int i  = 0; i < A_orig->local_num_rows; i++)
    {
        proc_part[i] = i % num_procs;
    } 
    ParCSRMatrix* A = repartition_matrix(A_orig, proc_part);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Partitioning Time %e\n", t0);

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);

    // TIME Original SpMV
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        A->mult(x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Partitioned SpMV Time %e\n", t0);

    // Time TAPSpMV
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_tests; i++)
    {
        A->tap_mult(x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_tests;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Partitioned TAPSpMV Time %e\n", t0);

    delete[] proc_part;
    delete A_orig;
    delete A;

    MPI_Finalize();
}

