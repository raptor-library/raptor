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
    ml = new ParMultilevel(A, strong_threshold, CLJP, Classical, SOR,
            1, 1.0, 50, -1);
    raptor_setup = MPI_Wtime() - t0;
    clear_cache(cache_array);

    // Solve Raptor Hierarchy
    x.set_const_value(0.0);
    std::vector<double> res;
    std::vector<double> level_times(ml->num_levels);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml->solve(x, b, res, level_times.data());
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

    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
        ParVector& xl = ml->levels[i]->x;

        if (rank == 0) printf("Level %d\n", i);

        int n_active;
        int num_msgs = Al->comm->send_data->num_msgs;
        if (Al->comm->recv_data->num_msgs > num_msgs)
            num_msgs = Al->comm->recv_data->num_msgs;
        int size_msgs = Al->comm->send_data->size_msgs;
        if (Al->comm->recv_data->size_msgs > size_msgs)
            size_msgs = Al->comm->recv_data->size_msgs;
        int has_comm = 0;
        if (num_msgs) has_comm = 1;
        MPI_Allreduce(&has_comm, &n_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        int num_nodes = num_procs / 16;
        int ppn = n_active / num_nodes;
        if (ppn == 0) ppn = 1;


        int n_short = 0;
        int n_eager = 0;
        int n_rend = 0;
        int s_short = 0;
        int s_eager = 0;
        int s_rend = 0;


        double short_a = 4.0e-6;
        double eager_a = 1.1e-5;
        double rend_a = 2.0e-5;
        double short_b_inj = 6.3e8;
        double short_b_max = -1.8e7;

        double eager_b_inj = 1.7e9;
        double eager_b_max = 6.2e7;

        double rend_b_inj = 3.6e9;
        double rend_b_max = 6.1e8;
        double rend_b_n = 5.5e9;

        double model = 0;
        for (int i = 0; i < Al->comm->send_data->num_msgs; i++)
        {
            int start = Al->comm->send_data->indptr[i];
            int end = Al->comm->send_data->indptr[i+1];
            int size = end - start;
            if (size < short_cutoff)
            {
                model += short_a + ((ppn * size) / (short_b_max + (ppn - 1)*short_b_inj));
            }
            else if (size < eager_cutoff)
            {
                model += eager_a + ((ppn * size) / (eager_b_max + (ppn - 1)*eager_b_inj));
            }
            else
            {
                double b_tmp = rend_b_max + (ppn-1)*rend_b_inj;
                if (rend_b_n < b_tmp)
                {
                    b_tmp = rend_b_n;
                }
                model += rend_a + ((ppn*size) / b_tmp);
            }
        }

        double comm_time = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < 100; i++)
        {
            Al->comm->communicate(xl);
        }
        comm_time = (MPI_Wtime() - t0) / 100;
        
        if (has_comm == 0) level_times[i] = 0.0;
        MPI_Reduce(&level_times[i], &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Level Time: %e\n", t0);
        MPI_Reduce(&model, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Model Time: %e\n", t0);
        MPI_Reduce(&comm_time, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Measured Comm Time: %e\n", t0);

        int reduced;
        if (rank == 0) printf("Active Procs: %d\n", n_active);
        MPI_Reduce(&num_msgs, &reduced, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max Num Msgs: %d\n", reduced);
        MPI_Reduce(&num_msgs, &reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Total Num Msgs: %d\n", reduced);
        MPI_Reduce(&size_msgs, &reduced, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Max Size Msgs: %d\n", reduced);
        MPI_Reduce(&size_msgs, &reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Total Size Msgs: %d\n", reduced);

    }


    // Delete raptor hierarchy
    delete ml;
    delete A;
    MPI_Finalize();

    return 0;
}

