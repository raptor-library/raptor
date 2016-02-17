#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "hypre_async.h"

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

        data_t* local_results;
        if (ml->b_list[i]->local_n)
        {
            local_results = new data_t[ml->b_list[i]->local_n];
        }

        if (ml->A_list[i]->local_rows)
        {
            data_t* x_data = ml->x_list[i]->local->data();
            for (int j = 0; j < ml->x_list[i]->local_n; j++)
            {
                x_data[j] = (1.0 * j) / ml->x_list[i]->local_n;
            }
        }
//        if (ml->A_list[i]->local_rows && ml->A_list[i]->offd_num_cols)
//        {
//            ml->A_list[i]->offd->convert(CSR);
//        }

//        MPI_Barrier(MPI_COMM_WORLD);
//        t0 = MPI_Wtime();
//        for (int j = 0; j < num_tests; j++)
//        {
//            parallel_spmv(ml->A_list[i], ml->x_list[i], ml->b_list[i], 1.0, 0.0, async);
//        }

        parallel_spmv(ml->A_list[i], ml->x_list[i], ml->b_list[i], 1.0, 0.0, 0);

        data_t* b_data = ml->b_list[i]->local->data();
        for (int j = 0; j < ml->b_list[i]->local_n; j++)
        {
            local_results[j] = b_data[j];
        }

        parallel_spmv(ml->A_list[i], ml->x_list[i], ml->b_list[i], 1.0, 0.0, 1);
        for (int j = 0; j < ml->b_list[i]->local_n; j++)
        {
            if (fabs(local_results[j] - b_data[j]) > zero_tol)
            {
                printf("Rank %d - Sync B[%d] = %2.3e, but ASYNC B[%d] = %2.3e\n", rank, j, local_results[j], j, b_data[j]);
            }
        }        

        if (ml->A_list[i]->local_rows && ml->A_list[i]->offd_num_cols)
        {
            ml->A_list[i]->offd->convert(CSR);
        }
        parallel_spmv(ml->A_list[i], ml->x_list[i], ml->b_list[i], 1.0, 0.0, 0);
        for (int j = 0; j < ml->b_list[i]->local_n; j++)
        {
            if (fabs(local_results[j] - b_data[j]) > zero_tol)
            {
                printf("Rank %d - Sync B[%d] = %2.3e, but CSR B[%d] = %2.3e\n", rank, j, local_results[j], j, b_data[j]);
            }
        }
    
        HYPRE_IJMatrix A_ij = convert(ml->A_list[i]);
        HYPRE_IJVector x_ij = convert(ml->x_list[i]);
        HYPRE_IJVector b_ij = convert(ml->b_list[i]);
    
        hypre_ParCSRMatrix* A_hypre;
        hypre_ParVector* x_hypre;
        hypre_ParVector* b_hypre;

        HYPRE_IJMatrixGetObject(A_ij, (void**) &A_hypre);
        HYPRE_IJVectorGetObject(x_ij, (void**) &x_hypre);
        HYPRE_IJVectorGetObject(b_ij, (void**) &b_hypre);

        hypre_ParCSRMatrixMatvec(1.0, A_hypre, x_hypre, 0.0, b_hypre);
        b_data = hypre_VectorData(hypre_ParVectorLocalVector(b_hypre));
        for (int j = 0; j < ml->b_list[i]->local_n; j++)
        {
            if (fabs(local_results[j] - b_data[j]) > zero_tol)
            {
                printf("Rank %d - Sync B[%d] = %2.3e, but HYPRE B[%d] = %2.3e\n", rank, j, local_results[j], j, b_data[j]);
            }
        }

        HYPRE_IJMatrixDestroy(A_ij);
        HYPRE_IJVectorDestroy(x_ij);
        HYPRE_IJVectorDestroy(b_ij);

//        tfinal = (MPI_Wtime() - t0) / num_tests;
//        MPI_Barrier(MPI_COMM_WORLD);

        // Print Timings and Info
//        local_nnz = 0;
//        global_nnz = 0;
//        if (ml->A_list[i]->local_rows)
//        {
//            local_nnz = ml->A_list[i]->diag->nnz + ml->A_list[i]->offd->nnz;
//        }
//        MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
//        if (rank == 0) printf("Level %d has %lu nonzeros\n", i, global_nnz);
//        double b_norm = ml->b_list[i]->norm(2);
//        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);

//        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//        if (rank == 0) printf("Level %d Max Time per (ASYNC=%d) SpMV: %2.3e\n", i, async, t0);
//        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//        if (rank == 0) printf("Level %d Avg Time per (ASYNC=%d) SpMV: %2.3e\n", i, async, t0 / num_procs);
    }

//    delete ml;


//    delete A;
//    delete x;
//    delete b;

    MPI_Finalize();

    return 0;
}



