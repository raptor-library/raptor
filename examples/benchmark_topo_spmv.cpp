#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/external/mfem_wrapper.hpp"
//#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/stencil.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/random.hpp"
//#include "gallery/noncontiguous.hpp"
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

    TopoManager* tm = new TopoManager();

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

    // Variables for Iso/Aniso Problems
    int n;
    int dim;
    int* grid;
    data_t* stencil;
    int nnz_per_row;
 
    data_t max_diff = 0.0;
    data_t total_max_diff;

    // Variable for IO 
    char* filename;

    // Variables for MFEM
    char* mesh;
    int num_elements;
    int order;
    int mfem_choice;

    // Timing variables
    double t0, t1;
    double tsync, tnode;
    double tsync_setup, tnode_setup;
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
            //filename = argv[2];
            //int sym = 1;
            //if (argc > 3) sym = atoi(argv[3]);
            //ParMatrix* Atmp = readParMatrix(filename, MPI_COMM_WORLD, 1, sym);
            //int* partition = ptscotch_partition(Atmp);
            //std::vector<coo_data> new_mat;
            //repartition_matrix(Atmp, new_mat, partition);
            //A = non_contiguous(Atmp->global_rows, Atmp->global_cols, new_mat);
            //delete[] partition;
            //delete Atmp;
            //
            //
            //A = readParMatrix(filename, MPI_COMM_WORLD, 1, sym);
            //int* partition = ptscotch_partition(A);
            //delete[] partition;
            

            n = 10000;
            nnz_per_row = 10;
            if (argc > 2) n = atoi(argv[2]);
            if (argc > 3) nnz_per_row = atoi(argv[3]);
            A = random_mat(n, n, nnz_per_row);
        }

        b = new ParVector(A->global_rows, A->local_rows, A->first_row);
        x = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);
        b->set_const_value(0.0);
        x->set_const_value(1.0);
    }
    else
    {
        //mesh = "/u/sciteam/bienz/mfem/data/beam-tet.mesh";
        mesh = "/Users/abienz/Documents/Parallel/mfem/data/escher.mesh";

        mfem_choice = 2;
        num_elements = 3;
        order = 3;
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
            mfem_laplace(&A, &x, &b, mesh, num_elements, order);
        }
        else if (mfem_choice == 1)
        {
            mfem_linear_elasticity(&A, &x, &b, mesh, num_elements, order);
        }
        else
        {
            mfem_darcy(&A, &x, &b, mesh, num_elements, order);
        }
    }

    data_t norm = 0.0;

    ParVector* b_tmp = new ParVector(A->global_rows, A->local_rows, A->first_row);
    for (int i = 0; i < b->local_n; i++)
    {
        b_tmp->local->data()[i] = b->local->data()[i];
    }

    for (int i = 0; i < x->local_n; i++)
    {
        x->local->data()[i] = A->first_row + i;
    }

    TopoComm* tc = new TopoComm(tm, A->first_row, A->first_col_diag, 
            A->local_to_global.data(), A->global_col_starts.data(), A->offd_num_cols);

    for (int test = 0; test < 5; test++)
    {
        if (rank == 0) printf("Test %d\n", test);

        delete tc;
        get_ctime(t0);
        tc = new TopoComm(tm, A->first_row, A->first_col_diag, A->local_to_global.data(),
            A->global_col_starts.data(), A->offd_num_cols);
        get_ctime(t1);
        tnode_setup = t1 - t0;

        delete A->comm;
        get_ctime(t0);
        A->comm = new ParComm(A->local_to_global, A->global_col_starts, A->first_row);
        get_ctime(t1);
        tsync_setup = t1 - t0;

        MPI_Reduce(&tnode_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Time Setup Node = %2.5e\n", t0);
        MPI_Reduce(&tsync_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Time Setup Sync = %2.5e\n", t0);

        tnode = 0.0;
        tsync = 0.0;

        int nnz = A->diag->nnz;
        if (A->offd_num_cols) nnz += A->offd->nnz;

        // Test Efficient SpMV
        MPI_Barrier(MPI_COMM_WORLD);
        for (int j = 0; j < num_tests; j++)
        {
            get_ctime(t0);
            parallel_spmv_topo(A, x, b_tmp, tc);
            get_ctime(t1);
            tnode += (t1 - t0);
            clear_cache(cache_len, cache_array);
        }
        tnode /= num_tests;
        norm = b_tmp->norm(2);
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

        data_t max_b = 0;
        data_t total_max_b = 0;
        for (int i = 0; i < b->local_n; i++)
        {
            data_t val = fabs(b->local->data()[i]);
            if (val > max_b) max_b = val;
        }
        MPI_Reduce(&max_b, &total_max_b, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        max_diff = 0;
        for (int i = 0; i < b->local_n; i++)
        {
            data_t diff = fabs(b->local->data()[i] - b_tmp->local->data()[i]);
            if (diff > max_diff) max_diff = diff;
        }
        MPI_Reduce(&max_diff, &total_max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max Difference = %e\n", total_max_diff);
        if (rank == 0) printf("Max Difference / Max B = %e\n", 
                total_max_diff / total_max_b);

            // Print Timing Information
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
       
    int orig_num_sends_l = 0;
    int orig_size_sends_l = 0;
    int new_num_sends_l = 0;
    int new_size_sends_l = 0;
       
    int send_start, send_end, send_proc;
    int hops;
    int coords1[tm->n_dims];
    int coords2[tm->n_dims];
    tc->tm->getCoordinates(rank, coords1);
    if (A->local_rows)
    {
        if (A->comm->num_sends)
        {
            for (int s = 0; s < A->comm->num_sends; s++)
            {
                send_start = A->comm->send_row_starts[s];
                send_end = A->comm->send_row_starts[s+1];
                send_proc = A->comm->send_procs[s];
                tc->tm->getCoordinates(send_proc, coords2);

                if (tm->sameNode(coords1, coords2))
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

            new_num_sends_l = tc->local_S_par_comm->num_sends
                    + tc->local_R_par_comm->num_sends
                    + tc->local_L_par_comm->num_sends;
            new_size_sends_l = tc->local_S_par_comm->size_sends
                    + tc->local_R_par_comm->size_sends
                    + tc->local_L_par_comm->size_sends;
                
            new_num_sends = tc->global_par_comm->num_sends;
            new_size_sends = tc->global_par_comm->size_sends;

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

    // Print Communication Data for Efficient SpMV
    MPI_Reduce(&new_num_sends, &msg_data, 1, MPI_INT,
            MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Max Num Msgs = %d\n", msg_data);
    MPI_Reduce(&new_num_sends, &msg_data, 1, MPI_INT, 
            MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Total Num Msgs = %d\n", msg_data);
    MPI_Reduce(&new_size_sends, &msg_data, 1, MPI_INT, 
            MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Max Size Msgs = %d\n", msg_data);
    MPI_Reduce(&new_size_sends, &msg_data, 1, MPI_INT, 
            MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("New - Total Size Msgs = %d\n", msg_data);

    MPI_Reduce(&new_num_sends_l, &msg_data, 1, MPI_INT,
            MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Local New - Max Num Msgs = %d\n", msg_data);
    MPI_Reduce(&new_num_sends_l, &msg_data, 1, MPI_INT, 
            MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Local New - Total Num Msgs = %d\n", msg_data);
    MPI_Reduce(&new_size_sends_l, &msg_data, 1, MPI_INT, 
            MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Local New - Max Size Msgs = %d\n", msg_data);
    MPI_Reduce(&new_size_sends_l, &msg_data, 1, MPI_INT, 
            MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Local New - Total Size Msgs = %d\n", msg_data);

    // Delete efficient communication package for level
    delete b_tmp;
    delete tc;

    // Clean up
    delete[] cache_array;

    delete A;
    delete x;
    delete b;

    delete tm;

    MPI_Finalize();

    return 0;
}



