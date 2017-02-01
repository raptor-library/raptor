#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "topo_aware/topo_comm.hpp"
#include "topo_aware/topo_spmv.hpp"
#include "topo_aware/topo_wrapper.hpp"
#include "clear_cache.hpp"
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

    int num_tests = 10;
    int num_elements = 10;
    char* mesh = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
    int order = 3;
    int mat = 0;
    if (argc > 1)
    {
        num_tests = atoi(argv[1]);
        if (argc > 2)
        {
            mesh = argv[2];
            if (argc > 3)
            {
                num_elements = atoi(argv[3]);
                if (argc > 4)
                {
                    order = atoi(argv[4]);
                    if (argc > 5)
                        mat = atoi(argv[5]);
                }
            }
        }
    }

    TopoManager* tm = new TopoManager();

    // Declare Variables
    ParMatrix* A;
    ParVector* x;
    ParVector* b;

    long local_nnz;
    long global_nnz;
    int len_b, len_x;
    int local_rows;

    if (mat == 0)
        mfem_linear_elasticity(&A, &x, &b, mesh, num_elements, order);
    else if (mat == 1)
        mfem_laplace(&A, &x, &b, mesh, num_elements, order);

    // Create hypre (amg_data) and raptor (ml) hierarchies (they share data)
    double t0, t1;
    double tsync, tnode;

    int cache_len = 10000;
    double* cache_array = new double[cache_len];

    data_t norm = 0.0;

    TopoComm* tc = new TopoComm(tm, A->first_row, A->first_col_diag, A->local_to_global.data(), A->global_col_starts.data(), A->offd_num_cols);
    x->set_rand_values();

    for (int test = 0; test < 5; test++)
    {
        tnode = 0.0;
        tsync = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        for (int j = 0; j < num_tests; j++)
        {
            get_ctime(t0);
            parallel_spmv_topo(A, x, b, tc);
            get_ctime(t1);
            tnode += (t1 - t0);

            clear_cache(cache_len, cache_array);
        }
        tnode /= num_tests;
        norm = b->norm(2);
        if (rank == 0) printf("Norm = %2.3e\n", norm);

        // Test CSC Synchronous SpMV
        MPI_Barrier(MPI_COMM_WORLD);
        for (int j = 0; j < num_tests; j++)
        {
            get_ctime(t0);
            parallel_spmv(A, x, b, 1.0, 0.0, 0);
            get_ctime(t1);
            tsync += (t1 - t0);

            clear_cache(cache_len, cache_array);
        }
        tsync /= num_tests;
        norm = b->norm(2);
        if (rank == 0) printf("Norm = %2.3e\n", norm);

        MPI_Reduce(&tnode, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Time Node = %2.5e\n", t0);

        MPI_Reduce(&tsync, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Time Sync = %2.5e\n", t0);
    }

    // Collect Communication Data
    int orig_num_sends = 0;
    int orig_size_sends = 0;
    int new_num_sends = 0;
    int new_size_sends = 0;

    if (A->local_rows)
    {
        orig_num_sends = A->comm->num_sends;
        orig_size_sends = A->comm->size_sends;
        new_num_sends = tc->global_par_comm->num_sends;
        new_size_sends = tc->global_par_comm->size_sends;
    }

    int msg_data = 0;
    MPI_Reduce(&orig_num_sends, &msg_data, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Orig - Max Num Msgs = %d\n", msg_data);

    MPI_Reduce(&orig_num_sends, &msg_data, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Orig - Total Num Msgs = %d\n", msg_data);

    MPI_Reduce(&orig_size_sends, &msg_data, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Orig - Max Size Msgs = %d\n", msg_data);

    MPI_Reduce(&orig_size_sends, &msg_data, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Orig - Total Size Msgs = %d\n", msg_data);

    MPI_Reduce(&new_num_sends, &msg_data, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Max Num Msgs = %d\n", msg_data);

    MPI_Reduce(&new_num_sends, &msg_data, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Total Num Msgs = %d\n", msg_data);

    MPI_Reduce(&new_size_sends, &msg_data, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Max Size Msgs = %d\n", msg_data);

    MPI_Reduce(&new_size_sends, &msg_data, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Total Size Msgs = %d\n", msg_data);

    delete tc;
    delete[] cache_array;
 
    delete A;
    delete x;
    delete b;
  
    delete tm;

    MPI_Finalize();

    return 0;
}



