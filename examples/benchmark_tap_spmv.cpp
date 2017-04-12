#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "gallery/par_random.hpp"

#include <assert.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    int n = 100;
    int m = 100;
    if (argc > 1)
    {
        n = atoi(argv[1]);
        if (argc > 2)
        {
            m = atoi(argv[2]);
        }
    }

    A = random_mat(n, m, 25);
    b = ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
    x = ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
    A->tap_comm = new TAPComm(A->off_proc_column_map,
            A->first_local_row, A->first_local_col, 
            A->global_num_cols, A->local_num_cols);

    for (int i = 0; i < x.local_n; i++)
    {
        x.local[i] = double(rand()) / RAND_MAX;
    }
    b.set_const_value(0.0);

    int n_tests = 500;
    double t0, tfinal;
    double t_par, t_tap;
    double b_norm, b_tap_norm;

    // Determine Setup Costs for ParComm and TAPComm
    for (int test = 0; test < 5; test++)
    {
        delete A->comm;
        A->comm = NULL;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        ParComm* comm = new ParComm(A->off_proc_column_map, 
                A->first_local_row, A->first_local_col,
                A->global_num_cols, A->local_num_cols);
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        delete A->tap_comm;
        A->tap_comm = NULL;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        A->tap_comm = new TAPComm(A->off_proc_column_map,
                A->first_local_row, A->first_local_col, 
                A->global_num_cols, A->local_num_cols);
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t_tap, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            printf("ParComm Setup Time = %e\nTAPComm Setup Time = %e\n", t_par, t_tap);
        }
    }
   
    // Determine SpMV vs TAPSpMV Costs
    for (int test = 0; test < 5; test++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            A->mult(x, b);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        b_norm = b.norm(2);
        MPI_Reduce(&tfinal, &t_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        b.set_const_value(0.0);
        
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_tests; i++)
        {
            A->tap_mult(x, b);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        b_tap_norm = b.norm(2);
        MPI_Reduce(&tfinal, &t_tap, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        b.set_const_value(0.0);

        if (rank == 0)
        {
            printf("ParSpMV Time = %e\nTAPSpMV Time = %e\n", t_par, t_tap);
            assert(fabs(b_norm - b_tap_norm) < zero_tol);
        }
    }

    delete A;
    MPI_Finalize();

    return 0;
}
