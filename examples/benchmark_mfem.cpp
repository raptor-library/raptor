#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "hypre_async.h"

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
    int num_elements = 10;
    if (argc > 1)
    {
        num_tests = atoi(argv[1]);
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
        }
    }


    // Declare Variables
    ParMatrix* A;
    ParVector* x;
    ParVector* b;
    Hierarchy* ml;
    ParMatrix* A_l;
    ParVector* x_l;
    ParVector* b_l;

    HYPRE_IJMatrix A_ij;
    HYPRE_IJVector x_ij;
    HYPRE_IJVector b_ij;
    hypre_ParCSRMatrix* A_hypre;
    hypre_ParVector* x_hypre;
    hypre_ParVector* b_hypre;
    hypre_CSRMatrix** offd_proc_list;

    long local_nnz;
    long global_nnz;
    index_t num_levels;
    index_t len_b, len_x;
    index_t local_rows;
    data_t b_norm;
    data_t t0, tfinal;
    data_t* b_data;
    data_t* x_data;

    //Initialize variable for clearing cache between tests
    index_t cache_size = 10000;
    data_t* cache_list = new data_t[cache_size];

    // Get matrix and vectors from MFEM
    //mfem_laplace(&A, &x, &b, mesh, num_elements, order);
    int dim = 3;
    int grid[dim] = {num_elements, num_elements, num_elements};
    data_t* sten = laplace_stencil_27pt();
    A = stencil_grid(sten, grid, dim);
    delete[] sten;
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

    // Create hypre (amg_data) and raptor (ml) hierarchies (they share data)
    A_ij = convert(A);
    x_ij = convert(x);
    b_ij = convert(b);
    HYPRE_IJMatrixGetObject(A_ij, (void**) &A_hypre);
    HYPRE_IJVectorGetObject(x_ij, (void **) &x_hypre);
    HYPRE_IJVectorGetObject(b_ij, (void **) &b_hypre);
    HYPRE_Solver amg_data = hypre_create_hierarchy(A_hypre, x_hypre, b_hypre, 6, 0, 0, 0, 0.25);
    ml = convert((hypre_ParAMGData*) amg_data);

    num_levels = ml->num_levels;
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray((hypre_ParAMGData*) amg_data);
    hypre_ParVector** x_array = hypre_ParAMGDataUArray((hypre_ParAMGData*) amg_data);
    hypre_ParVector** b_array = hypre_ParAMGDataFArray((hypre_ParAMGData*) amg_data);

    ml->x_list[0] = x;
    ml->b_list[0] = b;

    for (int i = 0; i < num_levels; i++)
    {
        A_l = ml->A_list[i];
        x_l = ml->x_list[i];
        b_l = ml->b_list[i];
        A_hypre = A_array[i];
        x_hypre = x_array[i];
        b_hypre = b_array[i];

        local_rows = A_l->local_rows;
        len_x = x_l->local_n;
        len_b = b_l->local_n;

        // Print Global Nonzeros in Level i
        if (local_rows)
        {
            local_nnz = A_l->diag->nnz + A_l->offd->nnz;
        }
        MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d has %lu nonzeros\n", i, global_nnz);

        // Set X Data -- Local Elements are Unique
        if (local_rows)
        {
            x_data = x_l->local->data();
            for (int j = 0; j < len_x; j++)
            {
                x_data[j] = (1.0 * j) / len_x;
            }
        }

        // Test CSC Synchronous SpMV
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 0);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        b_norm = b_l->norm(2);
        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Max Time per SYNC SpMV: %2.3e\n", i, t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Avg Time per SYNC SpMV: %2.3e\n", i, t0 / num_procs);
        clear_cache(cache_size, cache_list);   

        // Test CSC Asynchronous SpMV
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 1);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        b_norm = b_l->norm(2);
        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Max Time per ASYNC SpMV: %2.3e\n", i, t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Avg Time per ASYNC SpMV: %2.3e\n", i, t0 / num_procs);
        clear_cache(cache_size, cache_list);   

        // Test CSR Synchronous SpMV
        if (A_l->local_rows && A_l->offd_num_cols)
        {
            A_l->offd->convert(CSR);
        }
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 0);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        b_norm = b_l->norm(2);
        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Max Time per CSR SpMV: %2.3e\n", i, t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Avg Time per CSR SpMV: %2.3e\n", i, t0 / num_procs);
        clear_cache(cache_size, cache_list);   
    
        // Test HYPRE (CSR) Synchronous SpMV
        if (A_l->offd_num_cols)
        {
            hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A_hypre)) = A_l->offd->data.data();
            hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A_hypre)) = A_l->offd->indptr.data();
            hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A_hypre)) = A_l->offd->indices.data();
        }
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            hypre_ParCSRMatrixMatvec(1.0, A_hypre, x_hypre, 0.0, b_hypre);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        b_norm = sqrt(hypre_ParVectorInnerProd(b_hypre, b_hypre));
        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Max Time per HYPRE SpMV: %2.3e\n", i, t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Avg Time per HYPRE SpMV: %2.3e\n", i, t0 / num_procs);
        clear_cache(cache_size, cache_list);   
    
        // Test HYPRE (CSR) Asynchronous SpMV
        offd_proc_list = create_offd_proc_array(A_hypre);
        t0 = MPI_Wtime();
        for (int j = 0; j < num_tests; j++)
        {
            hypre_ParCSRMatrixAsyncMatvec(1.0, A_hypre, x_hypre, 0.0, b_hypre, offd_proc_list);
        }
        tfinal = (MPI_Wtime() - t0) / num_tests;
        b_norm = sqrt(hypre_ParVectorInnerProd(b_hypre, b_hypre));
        if (rank == 0) printf("2 norm of b = %2.3e\n", b_norm);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Max Time per HYPRE ASYNC SpMV: %2.3e\n", i, t0);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level %d Avg Time per HYPRE ASYNC SpMV: %2.3e\n", i, t0 / num_procs);
        clear_cache(cache_size, cache_list);   
    
        // Destroy HYPRE Matrices
        for (int j = 0; j < hypre_ParCSRCommPkgNumRecvs(hypre_ParCSRMatrixCommPkg(A_hypre)); j++)
        {
            hypre_CSRMatrixDestroy(offd_proc_list[j]);
        }
        delete[] offd_proc_list;
    }

    remove_shared_ptrs((hypre_ParAMGData*) amg_data);
    hypre_BoomerAMGDestroy(amg_data);
    HYPRE_IJMatrixDestroy(A_ij);
    HYPRE_IJVectorDestroy(x_ij);
    HYPRE_IJVectorDestroy(b_ij);

    delete ml;

    delete A;
    delete x;
    delete b;

    delete[] cache_list;

    MPI_Finalize();

    return 0;
}



