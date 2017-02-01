#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/random.hpp"
#include "topo_aware/topo_comm.hpp"
#include "topo_aware/topo_spmv.hpp"
#include "topo_aware/topo_emulator.hpp"
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

    // Variables for AMG hierarchy
    ParMatrix* A;
    ParVector* x;
    ParVector* b;
    ParVector* b_tmp;
 
    //int n_sizes = 6;
    //int sizes[6] = {1000, 5000, 10000, 15000, 20000, 25000};
    int n_sizes = 2;
    int sizes[2] = {10, 100};
    data_t density = 0.05;

    double t0, t1;
    double tsync, tnode;
    double tsync_setup, tnode_setup;
    int num_tests = 10;
    int max_diff, total_max_diff;

    // Variables to clear cache between tests
    int cache_len = 10000;
    double* cache_array = new double[cache_len];

    for (int i = 0; i < n_sizes; i++)
    {
        A = random_mat(sizes[i], sizes[i], density);
        b = new ParVector(A->global_rows, A->local_rows, A->first_row);
        b_tmp = new ParVector(A->global_rows, A->local_rows, A->first_row);
        x = new ParVector(A->global_cols, A->local_cols, A->first_col_diag);
        data_t norm = 0.0;
        for (int j = 0; j < b->local_n; j++)
        {
            b->local->data()[j] = 0;
            b_tmp->local->data()[j] = 0;
        }
        for (int j = 0; j < x->local_n; j++)
            x->local->data()[j] = 1.0;

        TopoComm* tc = new TopoComm(tm, A->first_row, A->first_col_diag, 
                A->local_to_global.data(),
                A->global_col_starts.data(), A->offd_num_cols);

        for (int test = 0; test < 5; test++)
        {
            if (rank == 0) printf("Test %d\n", test);

            delete tc;
            MPI_Barrier(MPI_COMM_WORLD);
            get_ctime(t0);
            tc = new TopoComm(tm, A->first_row, A->first_col_diag, A->local_to_global.data(),
                    A->global_col_starts.data(), A->offd_num_cols);
            get_ctime(t1);
            tnode_setup = t1 - t0;

            delete A->comm;
            MPI_Barrier(MPI_COMM_WORLD);
            get_ctime(t0);
            A->comm = new ParComm(A->local_to_global, A->global_col_starts, A->first_row);
            get_ctime(t1);
            tsync_setup = t1 - t0;

            MPI_Reduce(&tnode_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Size %d Time Setup Node = %2.5e\n", sizes[i], t0);
            MPI_Reduce(&tsync_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Size %d Time Setup Sync = %2.5e\n", sizes[i], t0);

            tnode = 0.0;
            tsync = 0.0;

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

       /*     data_t max_b = 0;
            data_t total_max_b = 0;
            for (int j = 0; j < b->local_n; j++)
            {
                data_t val = fabs(b->local->data()[j]);
                if (val > max_b) max_b = val;
            }
            MPI_Reduce(&max_b, &total_max_b, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            max_diff = 0;
            for (int j = 0; j < b->local_n; j++)
            {
                data_t diff = fabs(b->local->data()[j] - b_tmp->local->data()[j]);
                if (diff > max_diff) max_diff = diff;
            }
            MPI_Reduce(&max_diff, &total_max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Max Difference = %e\n", i, total_max_diff);
            if (rank == 0)
               if (fabs(total_max_b) < zero_tol)
                  printf("Level %d Max Difference / Max B = %e\n", i, total_max_diff / total_max_b);
                  */
            // Print Timing Information
            MPI_Reduce(&tnode, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Time Node = %2.5e\n", i, t0);
            MPI_Reduce(&tsync, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Level %d Time Sync = %2.5e\n", i, t0);
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
        int x0, y0, z0, t0;
        int x1, y1, z1, t1;
        tc->tm->getCoordinates(rank, &x0, &y0, &z0, &t0);
        if (A->local_rows)
        {
            if (A->comm->num_sends)
            {
                for (int s = 0; s < A->comm->num_sends; s++)
                {
                    send_start = A->comm->send_row_starts[s];
                    send_end = A->comm->send_row_starts[s+1];
                    send_proc = A->comm->send_procs[s];
                    tc->tm->getCoordinates(send_proc, &x1, &y1, &z1, &t1);

                    if (x1 == x0 && y1 == y0 && z1 == z0 && t1/PPN == t0/PPN)
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
        delete A;
        delete b;
        delete x;
        delete b_tmp;
        delete tc;
    }

    delete tm;

    // Clean up
    delete[] cache_array;
    delete tm;

    MPI_Finalize();

    return 0;
}


