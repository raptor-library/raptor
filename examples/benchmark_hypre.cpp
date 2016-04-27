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
#include "clear_cache.hpp"
#include <unistd.h>

#ifdef USE_AMPI
    #include "core/puppers.hpp"
#endif

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
    int aniso = 0;
    if (argc > 1)
    {
        num_tests = atoi(argv[1]);
        if (argc > 2)
        {
            num_elements = atoi(argv[2]);
            if (argc > 3)
            {
                aniso = atoi(argv[3]);
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
    data_t* b_data;
    data_t* x_data;

    // Get matrix and vectors from MFEM
    data_t* sten = NULL;
    int dim = 0;
    int* grid;
    if (aniso)
    {
        dim = 2;
        grid = new int[dim];
        for (int i = 0; i < dim; i++)
        {
            grid[i] = num_elements;
        }
        sten = diffusion_stencil_2d(0.01, M_PI/8.0);
    }
    else
    {
        dim = 3;
        grid = new int[dim];
        for (int i = 0; i < dim; i++)
        {
            grid[i] = num_elements;
        }
        sten = laplace_stencil_27pt();
    }
    A = stencil_grid(sten, grid, dim);
    delete[] sten;
    delete[] grid;
    b = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);
    x = new ParVector(A->global_rows, A->local_rows, A->first_row);
    x->set_const_value(1.0);

    // Create hypre (amg_data) and raptor (ml) hierarchies (they share data)
    int coarsen_type = 10;
    int interp_type = 6; 
    int Pmx = 0;
    int agg_num_levels = 1;

    if (aniso)
    {
        ml = create_wrapped_hierarchy(A, x, b);
    }
    else
    {
        ml = create_wrapped_hierarchy(A, x, b, coarsen_type, interp_type, Pmx, agg_num_levels);
    }

    int num_levels = ml->num_levels;
    Level* l0 = ml->levels[0];
    l0->x = x;
    l0->b = b;
    l0->has_vec = true;

    data_t t0, t1;
    data_t tsync[num_levels], tasync[num_levels];

    int num_lb = 2;
    data_t tsync_lb[num_levels][num_lb];
    data_t tasync_lb[num_levels][num_lb];

    int cache_len = 10000;
    double cache_array[cache_len];

#ifdef USE_AMPI
    int pup_idx;
    AMPI_Register_pup((MPI_PupFn) pup_hierarchy, &ml, &pup_idx);
    MPI_Info hints;
    MPI_Info_create(&hints);
    MPI_Info_set(hints, "ampi_load_balance", "sync");
#endif

    for (int i = 0; i < num_levels; i++)
    {
        Level* l = ml->levels[i];

        A_l = l->A;
        x_l = l->x;
        b_l = l->b;

        tsync[i] = 0.0;
        tasync[i] = 0.0;

        // Test CSC Synchronous SpMV
        for (int test = 0; test < num_tests; test++)
        {
            get_ctime(t0);
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 0);
            get_ctime(t1);
            tsync[i] += (t1 - t0);

            clear_cache(cache_len, cache_array);
        }
        tsync[i] /= num_tests;

        for (int test = 0; test < num_tests; test++)
        {
            get_ctime(t0);
            parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 1);
            get_ctime(t1);
            tasync[i] += (t1 - t0);

            clear_cache(cache_len, cache_array);
        }
        tasync[i] /= num_tests;
    }

    for (int lb = 0; lb < num_lb; lb++)
    {
#ifdef USE_AMPI
        AMPI_Migrate(hints);
        MPI_Barrier(MPI_COMM_WORLD);
#endif    

        for (int i = 0; i < ml->num_levels; i++)
        {
            Level* l = ml->levels[i];

            A_l = l->A;
            x_l = l->x;
            b_l = l->b;

            tsync_lb[i][lb] = 0.0;
            tasync_lb[i][lb] = 0.0;


            // Test CSC Synchronous SpMV
            for (int test = 0; test < num_tests; test++)
            {
                get_ctime(t0);
                parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 0);
                get_ctime(t1);
                tsync_lb[i][lb] += (t1 - t0);

                clear_cache(cache_len, cache_array);
            }
            tsync_lb[i][lb] /= num_tests;

            for (int test = 0; test < num_tests; test++)
            {
                get_ctime(t0);
                parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 1);
                get_ctime(t1);
                tasync_lb[i][lb] += (t1 - t0);

                clear_cache(cache_len, cache_array);
            }
            tasync_lb[i][lb] /= num_tests;
        }
    }

    for (int i = 0; i < num_levels; i++)
    {
        Level* l = ml->levels[i];

        A_l = l->A;
        x_l = l->x;
        b_l = l->b;

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
        MPI_Reduce(&tsync[i], &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tasync[i], &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            printf("Level %d\n", i);
            printf("Total Number of Messages Sent = %d\n", total_num_sends);
            printf("Total SIZE of Messages Sent = %d\n", total_size_sends);
            printf("Max Time per Parallel Spmv = %2.5e\n", t0);
            printf("Max Time per Parallel ASYNC SpMV = %2.5e\n", t1);
        }

        for (int lb = 0; lb < num_lb; lb++)
        {
            MPI_Reduce(&(tsync_lb[i][lb]), &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(tasync_lb[i][lb]), &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0)
            { 
                printf("Max Time per LOAD-BALANCED Parallel Spmv [%d] = %2.5e\n", lb, t0);
                printf("Max Time per LOAD-BALANCED Parallel ASYNC SpMV [%d] = %2.5e\n", lb, t1);
            }
        }
    }

    delete ml;

    MPI_Finalize();

    return 0;
}



