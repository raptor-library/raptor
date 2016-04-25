#include <mpi.h>
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
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
    int num_elements = 10;
    char* mesh = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
    int order = 3;
    int async = 0;
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
                    {
                        async = atoi(argv[5]);
                    }
                }
            }
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

    long local_nnz;
    long global_nnz;
    index_t len_b, len_x;
    index_t local_rows;
    data_t b_norm;
    data_t t0, tfinal, tfinal_min;
    data_t* b_data;
    data_t* x_data;

    // Get matrix and vectors from MFEM
//    mfem_linear_elasticity(&A, &x, &b, mesh, num_elements, order);
//mfem_laplace(&A, &x, &b, mesh, num_elements, order);
//mfem_electromagnetic_diffusion(&A, &x, &b, mesh, num_elements, order);
mfem_hdiv_diffusion(&A, &x, &b, mesh, num_elements, order);

    // Calculate and Print Number of Nonzeros in Matrix
    local_nnz = 0;
    if (A->local_rows)
    {
        local_nnz = A->diag->nnz + A->offd->nnz;
    }
    global_nnz = 0;
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Num Nonzeros = %lu\n", global_nnz);

    // Create hypre (amg_data) and raptor (ml) hierarchies (they share data)
    int coarsen_type = 10;
    int interp_type = 6; 
    int Pmx = 0;
    int agg_num_levels = 1;

//    ml = create_wrapped_hierarchy(A, x, b, coarsen_type, interp_type, Pmx, agg_num_levels);
    ml = create_wrapped_hierarchy(A, x, b);

    int num_levels = ml->num_levels;
    Level* l0 = ml->levels[0];
    l0->x = x;
    l0->b = b;
    l0->has_vec = true;

    for (int i = 0; i < num_levels; i++)
    {
        Level* l = ml->levels[i];

        A_l = l->A;
        x_l = l->x;
        b_l = l->b;

//        int l_reg = MPI_Register((void*) &l, (MPI_PupFn) pup_par_level);

//        MPI_Migrate();

        // Test CSC Synchronous SpMV
        tfinal_min = 10000;
MPI_Barrier(MPI_COMM_WORLD);
        num_tests = 0;
        t0 = MPI_Wtime();
        data_t total_time = 0.0;
        int finished = 0;
        int lcl_finished = 0;
        while(!finished)
        {
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, async);
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
                parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, async);
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
 
        if (A_l->local_rows)
        {
            num_sends = A_l->comm->num_sends;
            size_sends = A_l->comm->size_sends;
        }

        MPI_Reduce(&num_sends, &total_num_sends, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&size_sends, &total_size_sends, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            printf("Level %d\n", i);
            printf("Total Number of Messages Sent = %d\n", total_num_sends);
            printf("Total SIZE of Messages Sent = %d\n", total_size_sends);
            printf("Max Time per Parallel Spmv = %2.5e\n", t0);
        }
 

    }

    delete ml;

    delete A;
    delete x;
    delete b;


    MPI_Finalize();

    return 0;
}



