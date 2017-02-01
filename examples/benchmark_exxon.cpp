#include <mpi.h>
#include "timer.hpp"
#include <math.h>
#include "core/types.hpp"
#include "util/linalg/spmv.hpp"
#include "gallery/exxon_reader.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "gallery/matrix_IO.hpp"
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

    // Variables for AMG hierarchy
    ParMatrix* A;
    ParVector* x;
    ParVector* b;
    ParVector* b_tmp;
 
    // Variable for IO 
    char* folder;
    char* fname;
    char* iname = "index_R";
    char* suffix = ".bcoord_bin";

    if (argc > 1) folder = argv[1];
    else folder = "/u/sciteam/bienz/scratch/exxon_10_16/mat_1024";
 
    if (argc > 2) fname = argv[2];
    else fname = "spe10-1024-blk_coord_TS24_TSA0_NI0_R";

    if (argc > 3) suffix = argv[3];

    // Timing variables
    double t0, t1;
    double tsync, tnode, thypre;
    double tsync_setup, tnode_setup;
    int num_tests = 10;
    double max_b, total_max_b;
    double max_diff, total_max_diff;

    // Variables to clear cache between tests
    int cache_len = 10000;
    double* cache_array = new double[cache_len];

    int* global_rows;

    // Create System to be Solved
    if (rank == 0) printf("Reading file from partition %s\n", folder);
    A = exxon_reader(folder, iname, fname, suffix, &global_rows);
    x = new ParVector(A->global_rows, A->local_rows, A->first_row);
    b = new ParVector(A->global_rows, A->local_rows, A->first_row);
    b_tmp = new ParVector(A->global_rows, A->local_rows, A->first_row);
 
    for (int i = 0; i < x->local_n; i++)
        x->local->data()[i] = i;

    int nnz = A->diag->nnz;
    if (A->offd_num_cols) nnz += A->offd->nnz;
    int gbl_nnz;
    MPI_Reduce(&nnz, &gbl_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        printf("A->global_rows = %d\n", A->global_rows);
        printf("A->nnz = %d\n", gbl_nnz);
    }

    HYPRE_IJMatrix A_hypre = convert(A);
    HYPRE_IJVector x_hypre = convert(x);
    HYPRE_IJVector b_hypre = convert(b);

    hypre_ParCSRMatrix* parcsr_A;
    HYPRE_IJMatrixGetObject(A_hypre, (void**) &parcsr_A);
    hypre_ParVector* par_x;
    HYPRE_IJVectorGetObject(x_hypre, (void **) &par_x);
    hypre_ParVector* par_b;
    HYPRE_IJVectorGetObject(b_hypre, (void **) &par_b);

    double norm = 0.0;
    TopoComm* tc = new TopoComm(tm, A->first_row, A->first_col_diag, A->local_to_global.data(), A->global_col_starts.data(), A->offd_num_cols);
    for (int test = 0; test < 5; test++)
    {
        if (rank == 0) printf("Test %d\n", test);

        MPI_Barrier(MPI_COMM_WORLD);
        delete tc;
        get_ctime(t0);
        tc = new TopoComm(tm, A->first_row, A->first_col_diag, A->local_to_global.data(), A->global_col_starts.data(), 
                A->offd_num_cols);
        get_ctime(t1);
        tnode_setup = t1 - t0;

        MPI_Barrier(MPI_COMM_WORLD);
        delete A->comm;
        get_ctime(t0);
        A->comm = new ParComm(A->local_to_global, A->global_col_starts, A->first_row);
        get_ctime(t1);
        tsync_setup = t1 - t0;

        MPI_Reduce(&tnode_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tsync_setup, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("Exxon Matrix Time Setup Node = %e\n", t0);
            printf("Exxon Matrix Time Setup Sync = %e\n", t1);
        }
     
        tnode = 0.0;
        tsync = 0.0;
        thypre = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < num_tests; i++)
        {
            get_ctime(t0);
            parallel_spmv_topo(A, x, b_tmp, tc);
            get_ctime(t1);
            tnode += (t1 - t0);
            clear_cache(cache_len, cache_array);
        }
        tnode /= num_tests;
        norm = b_tmp->norm(2);
        if (rank == 0) printf("Exxon Matrix Node Norm = %e\n", norm);

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < num_tests; i++)
        {
            get_ctime(t0);
            parallel_spmv(A, x, b, 1.0, 0.0, 0);
            get_ctime(t1);
            tsync += (t1 - t0);
            clear_cache(cache_len, cache_array);
        }
        tsync /= num_tests;
        norm = b->norm(2);
        if (rank == 0) printf("Exxon Matrix Sync Norm = %e\n", norm);

        // Test HYPRE SpMV
        MPI_Barrier(MPI_COMM_WORLD);
        for (int j = 0; j < num_tests; j++)
        {
            get_ctime(t0);
            hypre_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_b);
            get_ctime(t1);
            thypre += (t1 - t0);
            clear_cache(cache_len, cache_array);
        }
        thypre /= num_tests;

        max_b = 0;
        total_max_b = 0;
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
        if (rank == 0)
        {
            printf("Exxon Matrix Max Difference = %e\n", total_max_diff);
            printf("Exxon Matrix Max Difference / Max B = %e\n", total_max_diff / total_max_b);
        }

        MPI_Reduce(&tnode, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Exxon Matrix Time Node = %e\n", t0);
        MPI_Reduce(&tsync, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Exxon Matrix Time Sync = %e\n", t0);
        MPI_Reduce(&thypre, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Exxon Matrix Time Hypre = %2.5e\n", t0);
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
    int dimx0, dimy0, dimz0, dimt0;
    int dimx, dimy, dimz, dimt;
    tc->tm->getCoordinates(rank, &dimx0, &dimy0, &dimz0, &dimt0);
    if (A->local_rows)
    {
        if (A->comm->num_sends)
        {
            for (int s = 0; s < A->comm->num_sends; s++)
            {
                send_start = A->comm->send_row_starts[s];
                send_end = A->comm->send_row_starts[s+1];
                send_proc = A->comm->send_procs[s];
                tc->tm->getCoordinates(send_proc, &dimx, &dimy, &dimz, &dimt);

                if (dimx == dimx0 && dimy == dimy0 && dimz == dimz0 && dimt/PPN == dimt0/PPN)
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

    HYPRE_IJMatrixDestroy(A_hypre);
    HYPRE_IJVectorDestroy(x_hypre);
    HYPRE_IJVectorDestroy(b_hypre);

    // Clean up
    delete[] cache_array;

    delete[] global_rows;
    delete A;
    delete x;
    delete b;

    delete tm;

    MPI_Finalize();

    return 0;
}


