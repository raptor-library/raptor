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
    double raptor_setup, raptor_solve;

    double strong_threshold = 0.25;
    int cache_len = 10000;
    std::vector<double> cache_array(cache_len);
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
        const char* file = "../../examples/LFAT5.mtx";
        int sym = 1;
        if (argc > 2)
        {
            file = argv[2];
            if (argc > 3)
            {
                sym = atoi(argv[3]);
            }
        }
        A = readParMatrix(file, MPI_COMM_WORLD, 1, sym);
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
    clear_cache(cache_array);

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParMultilevel(A, strong_threshold, RS, Direct, SOR,
            1, 1.0, 50, -1);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);

    int n_tests = 10;
    int comm;
    int proc, node;
    int start, end;
    int idx, size, recv_size;
    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParCSRMatrix* Pl = ml->levels[i]->P;
        if (!Al->tap_comm) Al->tap_comm = new TAPComm(Al->partition,
                Al->off_proc_column_map, Al->on_proc_column_map);

        if (rank == 0) printf("Level %d\n", i);

        // Time SpGEMM on Level i
        clear_cache(cache_array);
        tfinal = 0;
        for (int i = 0; i < n_tests; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            ParCSRMatrix* Cl = Al->mult(Pl);
            tfinal += MPI_Wtime() - t0;
            delete Cl;
            clear_cache(cache_array);
        }
        tfinal /= n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("SpGEMM Time: %e\n", t0);

        // Time TAP SpGEMM on Level i
        clear_cache(cache_array);
        tfinal = 0;
        for (int i = 0; i < n_tests; i++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            ParCSRMatrix* Cl = Al->tap_mult(Pl);
            tfinal += MPI_Wtime() - t0;
            delete Cl;
            clear_cache(cache_array);
        }
        tfinal /= n_tests;
        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP SpGEMM Time: %e\n", t0);


        // Gather number of messages / size (by inter/intra)
        int n_inter = 0;
        int n_intra = 0;
        int s_inter = 0;
        int s_intra = 0;
        int tap_n_inter = 0;
        int tap_n_intra = 0;
        int tap_s_inter = 0;
        int tap_s_intra = 0;

        int rank_node = Al->partition->topology->get_node(rank);
        for (int i = 0; i < Al->comm->send_data->num_msgs; i++)
        {
            proc = Al->comm->send_data->procs[i];
            start = Al->comm->send_data->indptr[i];
            end = Al->comm->send_data->indptr[i+1];
            node = Al->partition->topology->get_node(proc);
            size = 0;
            for (int j = start; j < end; j++)
            {
                idx = Al->comm->send_data->indices[j];
                size += (Pl->on_proc->idx1[idx+1] - Pl->on_proc->idx1[idx]) + 
                    (Pl->off_proc->idx1[idx+1] - Pl->off_proc->idx1[idx]);
            }
            if (node == rank_node)
            {
                n_intra += 2;
                s_intra += (end - start)*sizeof(int) + size*sizeof(int) + size*sizeof(double);
            }
            else
            {
                n_inter += 2;
                s_inter += (end - start)*sizeof(int) + size*sizeof(int) + size*sizeof(double);
            }
        }

        MPI_Reduce(&n_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Intra Msgs: %d\n", comm);
        MPI_Reduce(&n_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Num Inter Msgs: %d\n", comm);
        MPI_Reduce(&s_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Size Intra Msgs: %d\n", comm);
        MPI_Reduce(&s_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Size Inter Msgs: %d\n", comm);


        // Calculate message info from fully local par comm
        for (int i = 0; i < Al->tap_comm->local_L_par_comm->send_data->num_msgs; i++)
        {
            proc = Al->tap_comm->local_L_par_comm->send_data->procs[i];
            start = Al->tap_comm->local_L_par_comm->send_data->indptr[i];
            end = Al->tap_comm->local_L_par_comm->send_data->indptr[i+1];
            size = 0;
            for (int j = start; j < end; j++)
            {
                idx = Al->tap_comm->local_L_par_comm->send_data->indices[j];
                recv_size = (Pl->on_proc->idx1[idx+1] - Pl->on_proc->idx1[idx]) +
                    (Pl->off_proc->idx1[idx+1] - Pl->off_proc->idx1[idx]);
                size += recv_size;
            }
            tap_n_intra += 2;
            tap_s_intra += (end - start)*sizeof(int) + size*sizeof(int) + size*sizeof(double);
        }

        // Calculate message info from local_S par comm and send
        // communication sizes through communicator
        for (int i = 0; i < Al->tap_comm->local_S_par_comm->send_data->num_msgs; i++)
        {
            proc = Al->tap_comm->local_S_par_comm->send_data->procs[i];
            start = Al->tap_comm->local_S_par_comm->send_data->indptr[i];
            end = Al->tap_comm->local_S_par_comm->send_data->indptr[i+1];
            size = 0;
            for (int j = start; j < end; j++)
            {
                idx = Al->tap_comm->local_S_par_comm->send_data->indices[j];
                recv_size = (Pl->on_proc->idx1[idx+1] - Pl->on_proc->idx1[idx]) +
                    (Pl->off_proc->idx1[idx+1] - Pl->off_proc->idx1[idx]);
                Al->tap_comm->local_S_par_comm->send_data->int_buffer[j] = recv_size;
                size += recv_size;
            }
            tap_n_intra += 2;
            tap_s_intra += (end - start)*sizeof(int) + size*sizeof(int) + size*sizeof(double);
            MPI_Isend(&(Al->tap_comm->local_S_par_comm->send_data->int_buffer[start]), end - start, MPI_INT,
                    proc, Al->tap_comm->local_S_par_comm->key, Al->partition->topology->local_comm, 
                    &(Al->tap_comm->local_S_par_comm->send_data->requests[i]));
        }
        for (int i = 0; i < Al->tap_comm->local_S_par_comm->recv_data->num_msgs; i++)
        {
            proc = Al->tap_comm->local_S_par_comm->recv_data->procs[i];
            start = Al->tap_comm->local_S_par_comm->recv_data->indptr[i];
            end = Al->tap_comm->local_S_par_comm->recv_data->indptr[i+1];
            MPI_Irecv(&(Al->tap_comm->local_S_par_comm->recv_data->int_buffer[start]), end - start, MPI_INT,
                    proc, Al->tap_comm->local_S_par_comm->key, Al->partition->topology->local_comm,
                    &(Al->tap_comm->local_S_par_comm->recv_data->requests[i]));
        }
        if (Al->tap_comm->local_S_par_comm->send_data->num_msgs)
        {
            MPI_Waitall(Al->tap_comm->local_S_par_comm->send_data->num_msgs,
                    Al->tap_comm->local_S_par_comm->send_data->requests.data(),
                    MPI_STATUSES_IGNORE);
        }
        if (Al->tap_comm->local_S_par_comm->recv_data->num_msgs)
        {
            MPI_Waitall(Al->tap_comm->local_S_par_comm->recv_data->num_msgs,
                    Al->tap_comm->local_S_par_comm->recv_data->requests.data(),
                    MPI_STATUSES_IGNORE);
        }

        // Calculate message info from global par comm and send
        // communication sizes through communicator
        for (int i = 0; i < Al->tap_comm->global_par_comm->send_data->num_msgs; i++)
        {
            proc = Al->tap_comm->global_par_comm->send_data->procs[i];
            start = Al->tap_comm->global_par_comm->send_data->indptr[i];
            end = Al->tap_comm->global_par_comm->send_data->indptr[i+1];
            size = 0;
            for (int j = start; j < end; j++)
            {
                idx = Al->tap_comm->global_par_comm->send_data->indices[j];
                recv_size = Al->tap_comm->local_S_par_comm->recv_data->int_buffer[idx];
                Al->tap_comm->global_par_comm->send_data->int_buffer[j] = recv_size;
                size += recv_size;
            }
            tap_n_inter += 2;
            tap_s_inter += (end - start)*sizeof(int) + size*sizeof(int) + size*sizeof(double);
            MPI_Isend(&(Al->tap_comm->global_par_comm->send_data->int_buffer[start]), end - start, MPI_INT,
                    proc, Al->tap_comm->global_par_comm->key, MPI_COMM_WORLD, 
                    &(Al->tap_comm->global_par_comm->send_data->requests[i]));
        }
        for (int i = 0; i < Al->tap_comm->global_par_comm->recv_data->num_msgs; i++)
        {
            proc = Al->tap_comm->global_par_comm->recv_data->procs[i];
            start = Al->tap_comm->global_par_comm->recv_data->indptr[i];
            end = Al->tap_comm->global_par_comm->recv_data->indptr[i+1];
            MPI_Irecv(&(Al->tap_comm->global_par_comm->recv_data->int_buffer[start]), end - start, MPI_INT,
                    proc, Al->tap_comm->global_par_comm->key, MPI_COMM_WORLD,
                    &(Al->tap_comm->global_par_comm->recv_data->requests[i]));
        }
        if (Al->tap_comm->global_par_comm->send_data->num_msgs)
        {
            MPI_Waitall(Al->tap_comm->global_par_comm->send_data->num_msgs,
                    Al->tap_comm->global_par_comm->send_data->requests.data(),
                    MPI_STATUSES_IGNORE);
        }
        if (Al->tap_comm->global_par_comm->recv_data->num_msgs)
        {
            MPI_Waitall(Al->tap_comm->global_par_comm->recv_data->num_msgs,
                    Al->tap_comm->global_par_comm->recv_data->requests.data(),
                    MPI_STATUSES_IGNORE);
        }

        // Calculate message info from local_R par comm
        for (int i = 0; i < Al->tap_comm->local_R_par_comm->send_data->num_msgs; i++)
        {
            start = Al->tap_comm->local_R_par_comm->send_data->indptr[i];
            end = Al->tap_comm->local_R_par_comm->send_data->indptr[i+1];
            for (int j = start; j < end; j++)
            {
                idx = Al->tap_comm->local_R_par_comm->send_data->indices[j];
                size += Al->tap_comm->global_par_comm->recv_data->int_buffer[idx];
            }
            tap_n_intra += 2;
            tap_s_intra += (end - start)*sizeof(int) + size*sizeof(int) + size*sizeof(double);
        }

        MPI_Reduce(&tap_n_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Num Intra Msgs: %d\n", comm);
        MPI_Reduce(&tap_n_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Num Inter Msgs: %d\n", comm);
        MPI_Reduce(&tap_s_intra, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Size Intra Msgs: %d\n", comm);
        MPI_Reduce(&tap_s_inter, &comm, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("TAP Size Inter Msgs: %d\n", comm);
    }


    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}


