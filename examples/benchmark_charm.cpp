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
#include "core/puppers.hpp"
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
    int async = 0;
    int level = 0;
    if (argc > 1)
    {
        num_tests = atoi(argv[1]);
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
            if (argc > 3)
            {
                async = atoi(argv[3]);
                if (argc > 4)
                {
                    level = atoi(argv[4]);
                }
            }
        }
    }

    int ids[8];
    char names[8][20];

    snprintf(names[0], 20, "SpMV %d", level);
    snprintf(names[1], 20, "DiagSpMV %d", level);
    snprintf(names[2], 20, "OffdSpMV %d", level);
    snprintf(names[3], 20, "Waitany (Recv) %d", level);
    snprintf(names[4], 20, "Waitall (Recv) %d", level);
    snprintf(names[5], 20, "Waitall (Send) %d", level);
    snprintf(names[6], 20, "Irecv %d", level);
    snprintf(names[7], 20, "Isend %d", level);

usleep(1000000);
traceEnd();

    for (int i = 0; i < 8; i++)
    {
        ids[i] = _TRACE_REGISTER_FUNCTION_NAME((char*) names[i]);
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
    data_t t0, tfinal;
    data_t* b_data;
    data_t* x_data;

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

    ml = create_wrapped_hierarchy(A, x, b);

traceBegin();
usleep(1000000);

    int num_levels = ml->num_levels;
    Level* l0 = ml->levels[0];
    l0->x = x;
    l0->b = b;
    l0->has_vec = true;

    int num_sends, size_sends, total_num_sends, total_size_sends;

    int l_reg = 0;
    Level* l = NULL;

    if (num_levels > level)
    {
        l = ml->levels[level];
        l_reg = MPI_Register((void*) &l, (MPI_PupFn) pup_par_level);
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
    if (num_levels > level)
    {
        traceFlushLog();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < 4; i++)
    {
        if (num_levels > level)
        {
            l = ml->levels[level];
            A_l = l->A;
            x_l = l->x;
            b_l = l->b;

            MPI_Migrate();

//            _TRACE_BEGIN_FUNCTION_NAME(names[0]);
            for (int j = 0; j < num_tests; j++)
            {
                parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, async, names);
            }
//            _TRACE_END_FUNCTION_NAME(names[0]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (num_levels > level)
        {
            usleep(1000000);
            traceFlushLog();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }


    num_sends = 0;
    size_sends = 0;
    local_nnz = 0;
    if (num_levels > level)
    {
        l = ml->levels[level];
        A_l = l->A;
        x_l = l->x;
        b_l = l->b;
   
        num_sends = A_l->comm->num_sends;
        size_sends = A_l->comm->size_sends;
        if (A_l->offd_num_cols)
        {
            local_nnz = A_l->diag->nnz + A_l->offd->nnz;
        }
        else
        {
            local_nnz = A_l->diag->nnz;
        }
    }

    MPI_Reduce(&num_sends, &total_num_sends, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&size_sends, &total_size_sends, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &global_nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Level %d has %lu global nonzeros\n", level, global_nnz);
        printf("Total Number of Messages Sent = %d\n", total_num_sends);
        printf("Total SIZE of Messages Sent = %d\n", total_size_sends);
    }

    delete ml;

    delete A;
    delete x;
    delete b;


    MPI_Finalize();

    return 0;
}



