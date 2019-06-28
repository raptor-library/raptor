// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

//#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_cg_timed.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc < 2)
    {
        printf("Usage: <timings>\n");
        exit(-1);
    }

    int timings = atoi(argv[1]);

    double start, stop;

    /*int grid[2] = {1000, 1000};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);*/
    
    FILE* f;
    const char* mfem_fn = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    ParCSRMatrix* A = readParMatrix(mfem_fn);

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals;

    aligned_vector<double> avg_times;
    aligned_vector<double> max_times;
    aligned_vector<double> min_times;

    for (int i = 0; i < 3; i++)
    {
        avg_times.push_back(0.0);
        max_times.push_back(0.0);
        min_times.push_back(0.0);
    }

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    /* ---------- CG Timings ---------- */

    for (int i = 0; i < timings; i++)
    {
        residuals.clear();
        x.set_const_value(0.0);
        CG(A, x, b, avg_times, residuals, 1e-05, 3);
    }

    for (int i = 0; i < avg_times.size(); i++)
    {
        avg_times[i] = avg_times[i] / timings;
    }

    MPI_Reduce(&(avg_times[0]), &(max_times[0]), avg_times.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&(avg_times[0]), &(min_times[0]), avg_times.size(), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) MPI_Reduce(MPI_IN_PLACE, &(avg_times[0]), avg_times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    else MPI_Reduce(&(avg_times[0]), &(avg_times[0]), avg_times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print timings
    if (rank == 0) 
    {
        printf("all_reduce  point-to-point  computation\n");
        printf("---------------------------------------\n");
        printf("%lg \t %lg \t %lg\n", max_times[0], max_times[1], max_times[2]);
        printf("%lg \t %lg \t %lg\n", min_times[0], min_times[1], min_times[2]);
        printf("%lg \t %lg \t %lg\n", avg_times[0]/num_procs, avg_times[1]/num_procs, avg_times[2]/num_procs);
    }
    
    //delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0; 
} // end of main() //
