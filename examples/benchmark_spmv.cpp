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

#ifdef USING_MFEM
#include "gallery/external/mfem_wrapper.hpp"
#endif

//using namespace raptor;
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

    int cache_len = 10000;
    int num_tests = 2;

    std::vector<double> cache_array(cache_len);

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
            double eps = 0.001;
            double theta = M_PI/8.0;
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
#ifdef USING_MFEM
    else if (system == 2)
    {
        char* mesh_file = argv[2];
        int order = 2;
        int seq_refines = 1;
        int par_refines = 1;
        if (argc > 3)
        {
            order = atoi(argv[3]);
            if (argc > 4)
            {
                seq_refines = atoi(argv[4]);
                if (argc > 5)
                {
                    par_refines = atoi(argv[5]);
                }
            }
        }
        A = mfem_linear_elasticity(x, b, mesh_file, order);
    }
#endif
    else if (system == 3)
    {
        const char* file = "../../test_data/rss_A0.pm";
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
        x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
        b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
        x.set_const_value(1.0);
        A->mult(x, b);
    }

    int n_spmvs = 100;
    for (int i = 0; i < num_tests; i++)
    {
        clear_cache(cache_array);
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        for (int i = 0; i < n_spmvs; i++)
        {
            A->mult(x, b);
        }
        tfinal = (MPI_Wtime() - t0) / n_spmvs;

        MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("Test %d Max SpMV Time: %e\n", i, t0);
    }

    delete A;

    MPI_Finalize();

    return 0;
}

