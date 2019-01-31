// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "raptor.hpp"

using namespace raptor;

void compare(Vector& b, ParVector& b_par)
{
    double b_norm = b.norm(2);
    double b_par_norm = b_par.norm(2);

    assert(fabs(b_norm - b_par_norm) < 1e-06);

    Vector& b_par_lcl = b_par.local;
    for (int i = 0; i < b_par.local_n; i++)
    {
        assert(fabs(b_par_lcl[i] - b[i+b_par.first_local]) < 1e-06);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    int* proc_part;
    double bnorm;

    const char* filename = "../../../../examples/LFAT5.mtx";
    if (argc > 1) filename = argv[1];
    int n_tests = 100;
    aligned_vector<int> new_local_rows;

    // Create RowWise Partition 
    ParCSRMatrix* A_orig = readParMatrix(filename, MPI_COMM_WORLD, 
            true, 1);
    ParVector x_orig(A_orig->global_num_rows, A_orig->local_num_rows, 
            A_orig->partition->first_local_row);
    ParVector b_orig(A_orig->global_num_rows, A_orig->local_num_rows, 
            A_orig->partition->first_local_row);
    A_orig->tap_comm = new TAPComm(A_orig->partition, A_orig->off_proc_column_map);
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        x_orig[i] = A_orig->on_proc_column_map[i];
    }

    for (int i = 0; i < 3; i++)
    {
        // TIME Original SpMV
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_tests; j++)
        {
            A_orig->mult(x_orig, b_orig);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        bnorm = b_orig.norm(2);
        if (rank == 0) printf("Orig SpMV Time %e, Bnorm = %e\n", t0, bnorm);

        // Time TAPSpMV
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_tests; j++)
        {
            A_orig->tap_mult(x_orig, b_orig);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        bnorm = b_orig.norm(2);
        if (rank == 0) printf("Orig TAPSpMV Time %e, Bnorm = %e\n", t0, bnorm);
    }



    // RoundRobin Partitioning
    proc_part = new int[A_orig->local_num_rows];
    for (int i = 0; i < A_orig->local_num_rows; i++)
    {
        proc_part[i] = i % num_procs;
    }
    ParCSRMatrix* A_rr = repartition_matrix(A_orig, proc_part, new_local_rows);
    ParVector x_rr(A_rr->global_num_rows, A_rr->local_num_rows, 
            A_rr->partition->first_local_row);
    ParVector b_rr(A_rr->global_num_rows, A_rr->local_num_rows, 
            A_rr->partition->first_local_row);
    A_rr->tap_comm = new TAPComm(A_rr->partition, A_rr->off_proc_column_map);
    for (int i = 0; i < A_rr->local_num_rows; i++)
    {
        x_rr[i] = new_local_rows[i];
    }
    delete[] proc_part;

    for (int i = 0; i < 3; i++)
    {
        // TIME RoundRobin Orig SpMV
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_tests; j++)
        {
            A_rr->mult(x_rr, b_rr);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        bnorm = b_rr.norm(2);
        if (rank == 0) printf("RoundRobin SpMV Time %e, Bnorm = %e\n", t0, bnorm);

        // Time RoundRobin TAPSpMV
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_tests; j++)
        {
            A_rr->tap_mult(x_rr, b_rr);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        bnorm = b_rr.norm(2);
        if (rank == 0) printf("RoundRobin TAPSpMV Time %e, Bnorm = %e\n", t0, bnorm);
    }


    // Time Graph Partitioning
    t0 = MPI_Wtime();
    proc_part = ptscotch_partition(A_orig);
    ParCSRMatrix* A = repartition_matrix(A_orig, proc_part, new_local_rows);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Partitioning Time %e\n", t0);

    delete[] proc_part;
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map);
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x[i] = new_local_rows[i];
    }

    for (int i = 0; i < 3; i++)
    {
        // TIME Original SpMV
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_tests; j++)
        {
            A->mult(x, b);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        bnorm = b.norm(2);
        if (rank == 0) printf("Partitioned SpMV Time %e, Bnorm = %e\n", t0, bnorm);

        // Time TAPSpMV
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int j = 0; j < n_tests; j++)
        {
            A->tap_mult(x, b);
        }
        tfinal = (MPI_Wtime() - t0) / n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        bnorm = b.norm(2);
        if (rank == 0) printf("Partitioned TAPSpMV Time %e, Bnorm = %e\n", t0, bnorm);
    }

    delete A_orig;
    delete A_rr;
    delete A;

    MPI_Finalize();
}

