// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "clear_cache.hpp"

#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "core/types.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/par_multilevel.hpp"

#define eager_cutoff 1000
#define short_cutoff 62

void print_times(double time, double time_comm, const char* name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0;
    MPI_Reduce(&time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time: %e\n", name, t0);
    MPI_Reduce(&time_comm, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time Comm: %e\n", name, t0);
}

void print_tap_times(double time, double time_comm, const char* name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    double t0;
    MPI_Reduce(&time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s TAP Time: %e\n", name, t0);
    MPI_Allreduce(&time_comm, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("%s TAP Comm Time: %e\n", name, t0);
}

void time_spgemm(ParCSRMatrix* A, ParCSRMatrix* P)
{
    if (!A->comm) A->comm = new ParComm(A->partition, 
            A->off_proc_column_map, A->on_proc_column_map);

    double time, time_comm;
    int n_tests = 10;
    int cache_len = 10000;
    aligned_vector<double> cache_array(cache_len);

    A->spgemm_data.time = 0;
    A->spgemm_data.comm_time = 0;
    A->comm->reset_comm_data();
    
    // Initial matmult (grab comm data)
    {
        clear_cache(cache_array);

        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->mult(P);
        delete C;
   
        A->comm->print_comm_data(false);
    }
    for (int i = 1; i < n_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->mult(P);
        delete C;
    }
    time = A->spgemm_data.time / n_tests;
    time_comm = A->spgemm_data.comm_time / n_tests;

    print_times(time, time_comm, "SpGEMM");
}

void time_tap_spgemm(ParCSRMatrix* A, ParCSRMatrix* P)
{
    double time, time_comm;
    int n_tests = 10;
    int cache_len = 10000;
    aligned_vector<double> cache_array(cache_len);

    if (A->tap_comm) delete A->tap_comm;
    A->tap_comm = new TAPComm(A->partition, 
            A->off_proc_column_map, A->on_proc_column_map);

    // Time TAP SpGEMM on Level i
    A->spgemm_data.tap_time = 0;
    A->spgemm_data.tap_comm_time = 0;
    A->tap_comm->reset_comm_data();

    // Initial matmult (grab comm data)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);

        ParCSRMatrix* C = A->tap_mult(P);
        delete C;
     
        A->tap_comm->print_comm_data(false);
    }
    for (int i = 1; i < n_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->tap_mult(P);
        delete C;
    }
    time = A->spgemm_data.tap_time / n_tests;
    time_comm = A->spgemm_data.tap_comm_time / n_tests;

    print_tap_times(time, time_comm, "SpGEMM");

    delete A->tap_comm;
    A->tap_comm = NULL;
}

void time_spgemm_T(ParCSRMatrix* A, ParCSCMatrix* P)
{
    if (!P->comm) P->comm = new ParComm(P->partition, 
            P->off_proc_column_map, P->on_proc_column_map);

    double time, time_comm;
    int n_tests = 10;
    int cache_len = 10000;
    aligned_vector<double> cache_array(cache_len);

    // Time SpGEMM on Level i
    A->spgemm_T_data.time = 0;
    A->spgemm_T_data.comm_time = 0;
    P->comm->reset_comm_T_data();
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->mult_T(P);
        delete C;

        P->comm->print_comm_T_data(false);

    }
    for (int i = 1; i < n_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->mult_T(P);
        delete C;
    }
    time = A->spgemm_T_data.time / n_tests;
    time_comm = A->spgemm_T_data.comm_time / n_tests;
    print_times(time, time_comm, "Transpose SpGEMM");
}

void time_tap_spgemm_T(ParCSRMatrix* A, ParCSCMatrix* P)
{
    if (P->tap_comm) delete P->tap_comm;
    P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map, 
            P->on_proc_column_map);

    double time, time_comm;
    int n_tests = 10;
    int cache_len = 10000;
    aligned_vector<double> cache_array(cache_len);

    // Time TAP SpGEMM on Level i
    A->spgemm_T_data.tap_time = 0;
    A->spgemm_T_data.tap_comm_time = 0;
    P->tap_comm->reset_comm_T_data();
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->tap_mult_T(P);
        delete C;
    
        P->tap_comm->print_comm_T_data(false);
    }
    for (int i = 1; i < n_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        ParCSRMatrix* C = A->tap_mult_T(P);
        delete C;
    }
    time = A->spgemm_T_data.tap_time / n_tests;
    time_comm = A->spgemm_T_data.tap_comm_time / n_tests;

    print_tap_times(time, time_comm, "Transpose SpGEMM");

    delete P->tap_comm;
    P->tap_comm = NULL;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    ParCSRMatrix* A;
    ParCSCMatrix* A2;
    ParCSRMatrix* AT;

    double t0, tfinal;
    double t0_comm, tfinal_comm;
    int n0, s0;
    int nfinal, sfinal;
    int n_tests = 10;

    const char* file = "../../examples/LFAT5.pm";
    if (argc > 1)
        file = argv[1];
    A = readParMatrix(file);

    AT = (ParCSRMatrix*) A->transpose();
    if (AT->comm == NULL)
    {
        AT->comm = new ParComm(AT->partition, AT->off_proc_column_map,
                AT->on_proc_column_map);
    }
    if (AT->tap_comm == NULL)
    {
        AT->tap_comm = new TAPComm(AT->partition, AT->off_proc_column_map,
                AT->on_proc_column_map);
    }
    time_spgemm(AT, A);
    time_tap_spgemm(AT, A);
    delete AT;

    A2 = new ParCSCMatrix(A);
    if (A2->comm == NULL)
    {
        A2->comm = new ParComm(A2->partition, A2->off_proc_column_map,
                A2->on_proc_column_map);
    }
    if (A2->tap_comm == NULL)
    {
        A2->tap_comm = new TAPComm(A2->partition, A2->off_proc_column_map,
                A2->on_proc_column_map);
    }
    time_spgemm_T(A, A2);
    time_tap_spgemm_T(A, A2);
    delete A2;

    // Delete raptor hierarchy
    delete A;

    MPI_Finalize();

    return 0;
}


