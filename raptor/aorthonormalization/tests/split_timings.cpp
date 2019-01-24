// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "aorthonormalization/par_cgs_timed.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int timings = 25;
    double start, stop;

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParBVector *Q1_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *Q2_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *P_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *T_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    Vector T(W_bvecs, W_bvecs);

    aligned_vector<double> avg_times;
    aligned_vector<double> max_times;
    aligned_vector<double> min_times;

    aligned_vector<int> ts = {5, 10, 25, 50};

    for (int i = 0; i < 3; i++)
    {
        avg_times.push_back(0.0);
        max_times.push_back(0.0);
        min_times.push_back(0.0);
    }

    for (int i = 0; i < ts.size(); i++)
    {
        int t = ts[i];

        /* ---------- BCGS 2 Timings ---------- */
        for (int i = 0; i < timings; i++)
        {
            BCGS(A, *Q1_par, *Q2_par, *P_par, avg_times);
        }
        for (int j = 0; j < avg_times.size(); j++)
        {
            avg_times[i] = avg_times[i] / timings;
        }
        MPI_Reduce(&(avg_times[0]), &(max_times[0]), avg_times.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&(avg_times[0]), &(min_times[0]), avg_times.size(), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) MPI_Reduce(MPI_IN_PLACE, &(avg_times[0]), avg_times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        else MPI_Reduce(&(avg_times[0]), &(avg_times[0]), avg_times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
        if (rank == 0) 
        {
            printf("all_reduce  point-to-point  computation\n");
            printf("---------------------------------------\n");
            printf("- BCGS 2 t%d -\n", t);
            printf("%lg \t %lg \t %lg\n", max_times[0], max_times[1], max_times[2]);
            printf("%lg \t %lg \t %lg\n", min_times[0], min_times[1], min_times[2]);
            printf("%lg \t %lg \t %lg\n", avg_times[0]/num_procs, avg_times[1]/num_procs, avg_times[2]/num_procs);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

        /* ---------- BCGS 1 Timings ---------- */
        /* ---------- CGS Timings ---------- */
    }

    delete[] stencil;
    delete A;

    MPI_Finalize();

    return 0; 
} // end of main() //
