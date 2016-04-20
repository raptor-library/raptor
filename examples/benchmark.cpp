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
    char* filename = "/Users/abienz/Documents/Parallel/raptor/examples/msc01440.mtx";
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
    data_t t0, tfinal, tfinal_min;
    data_t* b_data;
    data_t* x_data;

    A = readParMatrix(filename, MPI_COMM_WORLD, 1, 1);

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


    // Test CSC Synchronous SpMV
    MPI_Barrier(MPI_COMM_WORLD);
    num_tests = 0;
    t0 = MPI_Wtime();
    data_t total_time = 0.0;
    int finished = 0;
    int lcl_finished = 0;
    while(!finished)
    {
        parallel_spmv(A, x, b, 1.0, 0.0, async);
        num_tests++;
        total_time = MPI_Wtime() - t0;
        if (total_time > 1.0) lcl_finished = 1;
        MPI_Allreduce(&lcl_finished, &finished, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    num_tests *= 2;
 
    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < 5; t++)
    { 
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            parallel_spmv(A, x, b, 1.0, 0.0, async);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        if (tfinal < tfinal_min) tfinal_min = tfinal;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    tfinal = tfinal_min;
    
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


    parallel_spmv_T(A, b, x, 1.0, 0.0);
    b->set_const_value(0.0);
    data_t b_norm = b->norm(2);
    data_t x_norm = x->norm(2);

    if (rank == 0) printf("BNorm = %2.3e\tXNorm = %2.3e\n", b_norm, x_norm);
 
    delete A;
    delete x;
    delete b;

    MPI_Finalize();

    return 0;
}



