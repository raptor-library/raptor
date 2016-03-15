#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/matrix_IO.hpp"
#include "hypre_async.h"
//#include "core/puppers.hpp"
#include <unistd.h>

using namespace raptor;

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get Local Process Rank, Number of Processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Get Command Line Arguments (Must Have 5)
    // TODO -- Fix how we parse command line
    int num_tests = 10;
    char* filename = "~/scratch/delaunay_n18.mtx";
    int async = 0;
    if (argc > 1)
    {
        num_tests = atoi(argv[1]);
        if (argc > 2)
        {
            filename = argv[2];
            if (argc > 3)
            {
                async = atoi(argv[3]);
            }
        }
    }

    // Declare Variables
    ParMatrix* A;
    ParVector* x;
    ParVector* b;

    long local_nnz;
    long global_nnz;
    index_t len_b, len_x;
    index_t local_rows;
    data_t b_norm;
    data_t t0, tfinal;
    data_t* b_data;
    data_t* x_data;

    // Get matrix and vectors from MFEM
    //mfem_laplace(&A, &x, &b, mesh, num_elements, order);
    A = readParMatrix(filename, MPI_COMM_WORLD, 1, 0);
    b = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);
    x = new ParVector(A->global_rows, A->local_rows, A->first_row);
    x->set_const_value(1.0);

    // Calculate and Print Number of Nonzeros in Matrix
    local_nnz = 0;
    if (A->local_rows)
    {
        local_nnz = A->diag->nnz + A->offd->nnz;
    }
    global_nnz = 0;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Nonzeros = %lu\n", global_nnz);

    t0, tfinal;

    // Test CSC Synchronous SpMV
    t0 = MPI_Wtime();
    for (int j = 0; j < num_tests; j++)
    {
        parallel_spmv(A, x, b, 1.0, 0.0, async);
    }
    tfinal = (MPI_Wtime() - t0) / num_tests;

    int num_sends = 0;
    int size_sends = 0;
    int total_num_sends = 0;
    int total_size_sends = 0;
 
    if (A->local_rows)
    {
        num_sends = A->comm->num_sends;
        size_sends = A->comm->size_sends;
    }

    MPI_Reduce(&num_sends, &total_num_sends, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&size_sends, &total_size_sends, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Total Number of Messages Sent = %d\n", total_num_sends);
        printf("Total SIZE of Messages Sent = %d\n", total_size_sends);
        printf("Max Time per Parallel Spmv = %2.5e\n", t0);
    }
 
    delete A;
    delete x;
    delete b;


    MPI_Finalize();

    return 0;
}



