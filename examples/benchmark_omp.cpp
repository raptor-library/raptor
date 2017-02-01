#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv_omp.hpp"
#include "gallery/external/mfem_wrapper.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "gallery/matrix_IO.hpp"
#include "topo_aware/topo_comm.hpp"
#include "topo_aware/topo_spmv.hpp"
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

    // Get Command Line Argument (For type of system to solve)
    // Iso: 0, Aniso: 1, IO: 2, MFEM: 3
    int system = 0;
    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    // Variables for AMG hierarchy
    ParMatrix* A;
    ParVector* x;
    ParVector* b;
    Hierarchy* ml;
    Level* level;
    ParMatrix* A_l;
    ParVector* x_l;
    ParVector* b_l;
    int num_levels;

    // Variables for Iso/Aniso Problems
    int n;
    int dim;
    int* grid;
    data_t* stencil;

    // Variable for IO 
    char* filename;

    // Variables for MFEM
    char* mesh;
    int num_elements;
    int order;
    int mfem_choice;

    // Timing variables
    double t0, t1;
    double tomp;
    int num_tests = 10;

    // Variables to clear cache between tests
    int cache_len = 10000;
    double* cache_array = new double[cache_len];

    // Create System to be Solved
    if (system < 3)
    {
        if (system == 0)
        {
            n = 435;
            if (argc > 2) n = atoi(argv[2]);
            grid = new int[3];
            grid[0] = n;
            grid[1] = n;
            grid[2] = n;
            dim = 3;
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            n = 9051;
            if (argc > 2) n = atoi(argv[2]);
            grid = new int[2];
            grid[0] = n;
            grid[1] = n;
            dim = 2;
            stencil = diffusion_stencil_2d(0.1, 0.0);
        }

        if (system < 2)
        {
            A = stencil_grid(stencil, grid, dim);
            delete[] stencil;
            delete[] grid;
        }
        else
        {
            filename = argv[2];
            A = readParMatrix(filename, MPI_COMM_WORLD, 1, 1);
        }

        b = new ParVector(A->global_rows, A->local_rows, A->first_row);
        x = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);
        b->set_const_value(0.0);
        x->set_const_value(1.0);
    }
    else
    {
        //mesh = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
        mesh = "/Users/abienz/Documents/Parallel/mfem/data/beam-tet.mesh";

        mfem_choice = 0;
        num_elements = 0;
        order = 0;
        if (argc > 2)
        {
            mfem_choice = atoi(argv[2]);
            if (argc > 3)
            {
                num_elements = atoi(argv[3]);
                if (argc > 4)
                    order = atoi(argv[4]);
            }
        }
        if (mfem_choice == 0)
        {
            if (num_elements == 0) num_elements = 3;
            if (order == 0) order = 3;
            mfem_linear_elasticity(&A, &x, &b, mesh, num_elements, order);
        }
        else if (mfem_choice == 1)
        {
            if (num_elements == 0) num_elements = 50000;
            if (order == 0) order = 5;
            mfem_laplace(&A, &x, &b, mesh, num_elements, order);
        }
        else if (mfem_choice == 2)
        {
            if (num_elements == 0) num_elements = 50000;
            if (order == 0) order = 3;
            mfem_electromagnetic_diffusion(&A, &x, &b, mesh, num_elements, order);
        }
        else if (mfem_choice == 3)
        {
            num_elements = 50000;
            order = 3;
            //mfem_hdiv_diffusion(&A, &x, &b, mesh, num_elements, order); 
        }
    }

    // Create AMG Hierarchy
    if (system == 0)
    {
        ml = create_wrapped_hierarchy(A, x, b, 10, 6, 0, 1, 0.35);
    }
    else
    {
        ml = create_wrapped_hierarchy(A, x, b);
    }

    // Initialize Variables
    num_levels = ml->num_levels;
    ml->levels[0]->x = x;
    ml->levels[0]->b = b;
    ml->levels[0]->has_vec = true;
    delete A;

    TopoManager* tm = new TopoManager();
    
    // Time the SpMV on each level of hierarchy
    data_t norm = 0.0;
    for (int i = 0; i < ml->num_levels; i++)
    {
        level = ml->levels[i];
        A_l = level->A;
        b_l = level->b;
        x_l = level->x;

        //l->x->set_const_value(1.0);
        for (int j = 0; j < x_l->local_n; j++)
        {
            x_l->local->data()[j] = A_l->first_row + j;
        }

        // Create efficient communication package
        for (int test = 0; test < 5; test++)
        {
            tomp = 0.0;

            // Test CSC Synchronous SpMV
            MPI_Barrier(MPI_COMM_WORLD);
            for (int j = 0; j < num_tests; j++)
            {
                get_ctime(t0);
                parallel_spmv(A_l, x_l, b_l, 1.0, 0.0, 0);
                get_ctime(t1);
                tomp += (t1 - t0);
                clear_cache(cache_len, cache_array);
            }
            tomp /= num_tests;
            norm = b_l->norm(2);
            if (rank == 0) printf("Norm = %2.3e\n", norm);

            // Print Timing Information
            MPI_Reduce(&tomp, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Time with OMP = %2.5e\n", i, t0);
        }

        // Collect Communication Data
        int orig_num_sends = 0;
        int orig_size_sends = 0;
        int orig_num_sends_l = 0;
        int orig_size_sends_l = 0;
       
        int send_start, send_end, send_proc;
        int hops;
        int x0, y0, z0, t0;
        int x, y, z, t;
        tm->getCoordinates(rank, &x0, &y0, &z0, &t0);
        if (A_l->local_rows)
        {
            if (A_l->comm->num_sends)
            {
                for (int s = 0; s < A_l->comm->num_sends; s++)
                {
                    send_start = A_l->comm->send_row_starts[s];
                    send_end = A_l->comm->send_row_starts[s+1];
                    send_proc = A_l->comm->send_procs[s];
                    tm->getCoordinates(send_proc, &x, &y, &z, &t);

                    if (x == x0 && y == y0 && z == z0 && t/PPN == t0/PPN)
                    {
                        orig_num_sends_l++;
                        orig_size_sends_l += send_end - send_start;
                    }
                    else
                    {
                        orig_num_sends++;
                        orig_size_sends += send_end - send_start;
                    }
                }
            }
        }

        // Print Communication Data for Original SpMV
        int msg_data = 0;
        MPI_Reduce(&orig_num_sends, &msg_data, 1, MPI_INT, 
                MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Orig - Max Num Msgs = %d\n", msg_data);
        MPI_Reduce(&orig_num_sends, &msg_data, 1, MPI_INT, 
                MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Orig - Total Num Msgs = %d\n", msg_data);
        MPI_Reduce(&orig_size_sends, &msg_data, 1, MPI_INT, 
                MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Orig - Max Size Msgs = %d\n", msg_data);
        MPI_Reduce(&orig_size_sends, &msg_data, 1, MPI_INT, 
                MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Orig - Total Size Msgs = %d\n", msg_data);

        MPI_Reduce(&orig_num_sends_l, &msg_data, 1, MPI_INT, 
                MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Local Orig - Max Num Msgs = %d\n", msg_data);
        MPI_Reduce(&orig_num_sends_l, &msg_data, 1, MPI_INT, 
                MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Local Orig - Total Num Msgs = %d\n", msg_data);
        MPI_Reduce(&orig_size_sends_l, &msg_data, 1, MPI_INT, 
                MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Local Orig - Max Size Msgs = %d\n", msg_data);
        MPI_Reduce(&orig_size_sends_l, &msg_data, 1, MPI_INT, 
                MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Local Orig - Total Size Msgs = %d\n", msg_data);
    }

    // Clean up
    delete ml->levels[0]->x;
    delete ml->levels[0]->b;
    delete[] cache_array;
    delete ml;
    delete tm;

    MPI_Finalize();

    return 0;
}





