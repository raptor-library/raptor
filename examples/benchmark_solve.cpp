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

#define eager_cutoff 8000
#define short_cutoff 496

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

    double t0;
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
    clear_cache(cache_array);

    // Setup Raptor Hierarchy
    MPI_Barrier(MPI_COMM_WORLD);    
    t0 = MPI_Wtime();
    ml = new ParMultilevel(A, strong_threshold, RS, Direct, SOR,
            1, 1.0, 50, -1);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);

    // Solve Raptor Hierarchy
    x.set_const_value(0.0);
    std::vector<double> res;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml->solve(x, b, res);
    raptor_solve = MPI_Wtime() - t0;
    clear_cache(cache_array);

    long lcl_nnz;
    long nnz;
    if (rank == 0) printf("Level\tNumRows\tNNZ\n");
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        lcl_nnz = Al->local_nnz;
        MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("%d\t%d\t%ld\n", i, Al->global_num_rows, nnz);
    }   

    if (rank == 0)
    {
        for (int i = 0; i < res.size(); i++)
        {
            printf("Res[%d] = %e\n", i, res[i]);
        }
    }   

    MPI_Reduce(&raptor_setup, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Setup Time: %e\n", t0);
    MPI_Reduce(&raptor_solve, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Raptor Solve Time: %e\n", t0);

    for (int i = 0; i < ml->num_levels - 1; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParVector& xl = ml->levels[i]->x;

        double short_a_socket = 8.864564e-07;
        double short_rcb_socket = 1.101622e+09;
        double eager_a_socket = 1.047754e-06; 
        double eager_rcb_socket = 1.586973e+09;
        double rend_a_socket = 3.373955e-06;
        double rend_rcb_socket =  3.129063e+09;

        double short_a_node = 1.701463e-06;
        double short_rcb_node = 2.471809e+08;
        double eager_a_node = 2.358333e-06;
        double eager_rcb_node = 4.843898e+08;
        double rend_a_node = 5.009861e-06;
        double rend_rcb_node = 3.069830e+09;

        double short_a = 4.249670e-06;
        double short_rcb = 7.006915e+08;
        double eager_a = 1.337284e-05;
        double eager_rcb = 9.838950e+08;
        double eager_rci = 3.168782e+08;
        double rend_a = 9.430665e-06; 
        double rend_rcb = 2.007557e+09;
        double rend_rci = 1.231642e+09;
        double rend_rn = 6.254932e+09;

        double model_a = 0;
        double model_b = 0;
        double model = 0;

        int eager_active = 0;
        int rend_active = 0;
        int num_rend_active, num_eager_active;
        int rend_size = 0;
        int eager_size = 0;
        int total_rend_size, total_eager_size;

        int num_socket = 8;
        int rank_node = Al->partition->topology->get_node(rank);
        int rank_socket = Al->partition->topology->get_local_proc(rank) / num_socket;

        for (int i = 0; i < Al->comm->send_data->num_msgs; i++)
        {
            int proc = Al->comm->send_data->procs[i];
            int start = Al->comm->send_data->indptr[i];
            int end = Al->comm->send_data->indptr[i+1];
            int node = Al->partition->topology->get_node(proc);
            int socket = Al->partition->topology->get_local_proc(proc) / num_socket;

            int size = (end - start) * sizeof(double);

 if (size < short_cutoff)
            {
                if (node == rank_node)
                {
                    if (socket == rank_socket)
                    {
                        model_a += short_a_socket;
                        model_b += size / short_rcb_socket;
                    }
                    else
                    {
                        model_a += short_a_node;
                        model_b += size / short_rcb_node;
                    }
                }
                else
                {
                    model_a += short_a;
                    model_b += size / short_rcb;
                }
            }
            else if (size < eager_cutoff)
            {
                if (node == rank_node)
                {
                    if (socket == rank_socket)
                    {
                        model_a += eager_a_socket;
                        model_b += size / eager_rcb_socket;
                    }
                    else
                    {
                        model_a += eager_a_node;
                        model_b += size / eager_rcb_node;
                    }
                }
                else
                {
                    model_a += eager_a;
                    eager_active = 1;
                    eager_size += size;
                }
            }
            else
            {
                if (node == rank_node)
                {
                    if (socket == rank_socket)
                    {
                        model_a += rend_a_socket;
                        model_b += size / rend_rcb_socket;
                    }
                    else
                    {
                        model_a += rend_a_node;
                        model_b += size / rend_rcb_node;
                    }
                }
                else
                {
                    model_a += rend_a;
                    rend_active = 1;
                    rend_size += size;
                }
            }
        }

        MPI_Allreduce(&eager_active, &num_eager_active, 1, MPI_INT, MPI_SUM, 
                Al->partition->topology->local_comm);
        MPI_Allreduce(&rend_active, &num_rend_active, 1, MPI_INT, MPI_SUM, 
                Al->partition->topology->local_comm);
        MPI_Allreduce(&eager_size, &total_eager_size, 1, MPI_INT, MPI_SUM, 
                Al->partition->topology->local_comm);
        MPI_Allreduce(&rend_size, &total_rend_size, 1, MPI_INT, MPI_SUM, 
                Al->partition->topology->local_comm);
        if (eager_active)
        {
            model_b += eager_size / (eager_rcb + (num_eager_active*eager_rci));
        }
        if (rend_active)
        {
            double b_tmp = rend_rcb + (num_rend_active * rend_rci);
            if (rend_rn < b_tmp) b_tmp = rend_rn;
            model_b += rend_size / b_tmp;
        }
        model = model_a + model_b;


        double comm_time = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < 100; i++)
        {
            Al->comm->communicate(xl);
        }
        comm_time = (MPI_Wtime() - t0) / 100;
       

        MPI_Reduce(&ml->level_times[i], &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level Time: %e\n", t0);
        MPI_Reduce(&ml->spmv_times[i], &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("SpMV Time: %e\n", t0);
        MPI_Reduce(&ml->spmv_comm_times[i], &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("SpMV Comm Time: %e\n", t0);

        MPI_Reduce(&comm_time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Measured Comm Time: %e\n", t0);
        MPI_Reduce(&model, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Model Time: %e\n", t0);
        MPI_Reduce(&model_a, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Model Latency Time: %e\n", t0);
        MPI_Reduce(&model_b, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Model BW Time: %e\n", t0);
    }


    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}

