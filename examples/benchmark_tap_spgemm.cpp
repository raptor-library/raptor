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
#include "gallery/par_stencil.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "multilevel/par_multilevel.hpp"

#define eager_cutoff 1000
#define short_cutoff 62

void print_times(double time, double time_comm, double time_wait, const char* name)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0;
    MPI_Reduce(&time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time: %e\n", name, t0);
    MPI_Reduce(&time_comm, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time Comm: %e\n", name, t0);
    MPI_Reduce(&time_wait, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Time Wait: %e\n", name, t0);
}

void print_tap_times(double time, double time_comm, double* time_wait, const char* name,
        bool use_S = true)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    double t0;
    double max_time_wait[4];
    MPI_Reduce(&time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%s TAP Time: %e\n", name, t0);
    MPI_Allreduce(&time_comm, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("%s TAP Comm Time: %e\n", name, t0);
    if (fabs(time_comm - t0) > zero_tol)
    {
        for (int i = 0; i < 4; i++)
            time_wait[i] = 0;
    }
    MPI_Reduce(time_wait, max_time_wait, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        char letters[4] = {'L', 'R', 'S', 'G'};
        for (int i = 0; i < 4; i++)
        {
            if (i == 2 && !use_S) continue;
            printf("%s TAP Wait %c Time: %e\n", name, letters[i], max_time_wait[i]);
        }
    }
}

void time_spgemm(ParCSRMatrix* A, ParCSRMatrix* P)
{
    if (!A->comm) A->comm = new ParComm(A->partition, 
            A->off_proc_column_map, A->on_proc_column_map);

    double time, time_comm, time_wait;
    int n_tests = 10;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);

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
    time_wait = A->comm->send_data->matrix_data.wait_time / n_tests;

    print_times(time, time_comm, time_wait, "SpGEMM");
}

void time_tap_spgemm(ParCSRMatrix* A, ParCSRMatrix* P, bool use_S = true)
{
    double time, time_comm;
    double time_wait[4];
    int n_tests = 10;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);

    if (A->tap_comm) delete A->tap_comm;
    A->tap_comm = new TAPComm(A->partition, 
            A->off_proc_column_map, A->on_proc_column_map, use_S);

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

    if (use_S)
    {        
        time_wait[2] = A->tap_comm->local_S_par_comm->send_data->matrix_data.wait_time / n_tests;
    }

    time_wait[0] = A->tap_comm->local_L_par_comm->send_data->matrix_data.wait_time / n_tests;
    time_wait[1] = A->tap_comm->local_R_par_comm->send_data->matrix_data.wait_time / n_tests;
    time_wait[3] = A->tap_comm->global_par_comm->send_data->matrix_data.wait_time / n_tests;
 
    print_tap_times(time, time_comm, time_wait, "SpGEMM0", use_S);

    delete A->tap_comm;
    A->tap_comm = NULL;
}

void time_spgemm_T(ParCSRMatrix* A, ParCSCMatrix* P)
{
    if (!P->comm) P->comm = new ParComm(P->partition, 
            P->off_proc_column_map, P->on_proc_column_map);

    double time, time_comm, time_wait;
    int n_tests = 10;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);

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
    time_wait = P->comm->recv_data->matrix_data.wait_time / n_tests;
    print_times(time, time_comm, time_wait, "Transpose SpGEMM");
}

void time_tap_spgemm_T(ParCSRMatrix* A, ParCSCMatrix* P, bool use_S = true)
{
    if (P->tap_comm) delete P->tap_comm;
    P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map, 
            P->on_proc_column_map, use_S);

    double time, time_comm;
    double time_wait[4];
    int n_tests = 10;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);

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
    if (use_S)
    {
        time_wait[2] = P->tap_comm->local_S_par_comm->recv_data->matrix_data.wait_time / n_tests;
    }
    time_wait[0] = P->tap_comm->local_L_par_comm->recv_data->matrix_data.wait_time / n_tests;
    time_wait[1] = P->tap_comm->local_R_par_comm->recv_data->matrix_data.wait_time / n_tests;
    time_wait[3] = P->tap_comm->global_par_comm->recv_data->matrix_data.wait_time / n_tests;

    print_tap_times(time, time_comm, time_wait, "Transpose SpGEMM", use_S);

    delete P->tap_comm;
    P->tap_comm = NULL;
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim;
    int n = 5;
    int system = 0;

    if (argc > 1)
    {
        system = atoi(argv[1]);
    }

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;
    double t0_comm, tfinal_comm;
    double t0_wait, tfinal_wait;
    double wait_L, wait_R, wait_S, wait_G;
    int n0, s0;
    int nfinal, sfinal;
    double raptor_setup, raptor_solve;

    double strong_threshold = 0.25;
    std::vector<double> residuals;

    if (system < 2)
    {
        double* stencil = NULL;
        std::vector<int> grid;
        if (argc > 2)
        {
            n = atoi(argv[2]);
        }

        if (system == 0)
        {
            dim = 3;
            grid.resize(dim, n);
            stencil = laplace_stencil_27pt();
        }
        else if (system == 1)
        {
            dim = 2;
            grid.resize(dim, n);
            double eps = 0.1;
            double theta = M_PI/4.0;
            if (argc > 3)
            {
                eps = atof(argv[3]);
                if (argc > 4)
                {
                    theta = atof(argv[4]);
                }
            }
            stencil = diffusion_stencil_2d(eps, theta);
        }
        A = par_stencil_grid(stencil, grid.data(), dim);
        delete[] stencil;
    }
    else if (system == 3)
    {
        const char* file = "../../examples/LFAT5.pm";
        A = readParMatrix(file);
    }

    if (system != 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map,
                A->on_proc_column_map);
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_rand_values();
        A->mult(x, b);
    }

    ParMultilevel* ml;

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParMultilevel(A, strong_threshold, RS, Direct, SOR,
            1, 1.0, 50, -1);
    raptor_setup = MPI_Wtime() - t0;

    ParCSRMatrix* A3 = ml->levels[3]->A;
    if (A3->tap_comm) delete A3->tap_comm;
    A3->tap_comm = new TAPComm(A3->partition, A3->off_proc_column_map,
            A3->on_proc_column_map, false);
    
    /*for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        ParCSCMatrix* Pl_csc = new ParCSCMatrix(Pl);
        ParCSRMatrix* AP = Al->mult(Pl);

        if (rank == 0) printf("Level %d\n", i);

        if (rank == 0) printf("A*P:\n");
        time_spgemm(Al, Pl);
        
        if (rank == 0) printf("\nTAP A*P:\n");
        time_tap_spgemm(Al, Pl);

        if (rank == 0) printf("\nSimple TAP A*P:\n");
        time_tap_spgemm(Al, Pl, false);

        if (rank == 0) printf("\nP.T*AP:\n");
        time_spgemm_T(AP, Pl_csc);

        if (rank == 0) printf("\nTAP P.T*AP:\n");
        time_tap_spgemm_T(AP, Pl_csc);

        if (rank == 0) printf("\nSimple TAP P.T*AP:\n");
        time_tap_spgemm_T(AP, Pl_csc, false);

        delete Pl_csc;
        delete AP;

        if (rank == 0) printf("Now testing P_new...\n");
        ParCSRMatrix* P_new = Al->mult(Pl);
        ParCSRMatrix* AP_new = Al->mult(P_new);
        ParCSCMatrix* P_new_csc = new ParCSCMatrix(P_new);

        if (rank == 0) printf("A*P:\n");
        time_spgemm(Al, P_new);
        
        if (rank == 0) printf("\nTAP A*P:\n");
        time_tap_spgemm(Al, P_new);

        if (rank == 0) printf("\nP.T*AP:\n");
        time_spgemm_T(AP_new, P_new_csc);

        if (rank == 0) printf("\nTAP P.T*AP:\n");
        time_tap_spgemm_T(AP_new, P_new_csc);        
        delete P_new;
        delete P_new_csc;
        delete AP_new;
    }
*/

    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}


