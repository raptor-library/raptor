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

#ifdef USE_AMPI
    MPI_Info hints;
    MPI_Info_create(&hints);
    MPI_Info_set(hints, "ampi_load_balance", "sync");
#endif

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

    long local_nnz;
    long global_nnz;
    int len_b, len_x;
    int local_rows;

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
    delete A;

    int num_levels = ml->num_levels;
    Level* l = ml->levels[0];
    l->x = x;
    l->b = b;
    l->has_vec = true;

    double t0, t1;
    double tsync, tasync;

    int num_lb = 2;

    int cache_len = 10000;
    double cache_array[cache_len];

#ifdef USE_AMPI
    int ml_idx;
    AMPI_Register_pup((MPI_PupFn) pup_hierarchy, &ml, &ml_idx);
#endif


    for (int lb = 0; lb < num_lb; lb++)
    {
        if (rank == 0)
        {
            printf("Number of Load Balancing Calls = %d\n", lb);
        }
        for (int i = 0; i < ml->num_levels; i++)
        {
            Level* l = ml->levels[i];

            tsync = 0.0;
            tasync = 0.0;

            // Test CSC Synchronous SpMV
            for (int j = 0; j < num_tests; j++)
            {
                get_ctime(t0);
                parallel_spmv(l->A, l->x, l->b, 1.0, 0.0, 0);
                get_ctime(t1);
                tsync += (t1 - t0);

                clear_cache(cache_len, cache_array);
            }
            tsync /= num_tests;

            for (int j = 0; j < num_tests; j++)
            {
                get_ctime(t0);
                parallel_spmv(l->A, l->x, l->b, 1.0, 0.0, 1);
                get_ctime(t1);
                tasync += (t1 - t0);

                clear_cache(cache_len, cache_array);
            }
            tasync /= num_tests;

            MPI_Reduce(&tsync, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&tasync, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) 
            {
                printf("Level %d Time Sync = %2.5e\n", i, t0);
                printf("Level %d Time Async = %2.5e\n", i, t1);
            }
        }
#ifdef USE_AMPI
        AMPI_Migrate(hints);
#endif
    }

    delete ml->levels[0]->x;
    delete ml->levels[0]->b;

    delete ml;

    MPI_Finalize();

    return 0;
}



