// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "raptor.hpp"

using namespace raptor;

int argc;
char **argv;

int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (_argc < 2)
    {
        printf("Usage: <nrhs> <nap_version>\n");
        exit(-1);
    }
    
    setenv("PPN", "4", 1);

    // Grab command line arguments
    int nrhs = atoi(_argv[1]);
    int nap_version = atoi(_argv[2]);

    // Setup matrix and vectors
    int grid[2] = {5000, 5000};
    //int grid[2] = {500, 500};
    double eps = 0.001;
    double theta = M_PI / 8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    delete[] stencil;

    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    aligned_vector<double> residuals;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Setup for Timings
    bool tap_comm = false;
    double* comp_time = new double[1];
    double* aort_time = new double[1];
    init_profile();
    *comp_time = 0.0;
    *aort_time = 0.0;
    
    // Setup correct node aware communication
    if (nap_version == 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
        tap_comm = true;
    }
    else if (nap_version == 3) 
    {
        // tap comm will get created when first tap aware spmv called
        tap_comm = true;
    }

    // Call SRECG
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    SRECG(A, x, b, nrhs, residuals, 1e-5, A->global_num_rows, comp_time, aort_time, tap_comm);

    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("SRECG");
    double t;
    MPI_Allreduce(comp_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("Computation Time: %e\n", t);
    MPI_Allreduce(aort_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    
    if (rank == 0 && t > 0.0) printf("Aortho Computation Time: %e\n", t);
    
    setenv("PPN", "16", 1);
 
    delete A;
    delete comp_time;
    delete aort_time;
    MPI_Finalize();
    return 0;

} // end of main() //
