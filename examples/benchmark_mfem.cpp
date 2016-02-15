#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"

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

//    mfem_linear_elasticity(&A, &x, &b, mesh, num_elements, order);
    mfem_laplace(&A, &x, &b, mesh, num_elements, order);

    x->set_const_value(1.0);
    b->set_const_value(0.0);

    long local_nnz = A->diag->nnz + A->offd->nnz;
    long global_nnz;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Nonzeros = %lu\n", global_nnz);

    data_t t0, tfinal;
    Hierarchy* ml = create_wrapped_hierarchy(A, x, b);
    int num_levels = ml->num_levels;

    ml->x_list[0] = x;
    ml->b_list[0] = b;

    for (int i = 0; i < num_levels; i++)
    {
        data_t* x_data = ml->x_list[i]->local->data();
        for (int j = 0; j < ml->x_list[i]->local_n; j++)
        {
            x_data[j] = (ml->A_list[i]->first_row + j) / (1.0 * ml->A_list[i]->global_rows);
        }
//        if (ml->A_list[i]->local_rows && ml->A_list[i]->offd_num_cols)
//        {
//            ml->A_list[i]->offd->convert(CSR);
//        }
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            parallel_spmv(ml->A_list[i], ml->x_list[i], ml->b_list[i], 1.0, 0.0, async);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        MPI_Barrier(MPI_COMM_WORLD);

        // Print Timings and Info
        long level_nnz = 0;
        long level_nnz_local = 0;
        if (ml->A_list[i]->local_rows)
        {
            level_nnz_local = ml->A_list[i]->diag->nnz + ml->A_list[i]->offd->nnz;
        }
        MPI_Reduce(&level_nnz_local, &level_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %lu has %d nonzeros\n", i, level_nnz);
        double b_norm = ml->b_list[i]->norm(2);
        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);

        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Max Time per (ASYNC=%d) SpMV: %2.3e\n", i, async, t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Avg Time per (ASYNC=%d) SpMV: %2.3e\n", i, async, t0 / num_procs);
    }

    delete ml;
    delete A;
    delete x;
    delete b;

    MPI_Finalize();

    return 0;
}



