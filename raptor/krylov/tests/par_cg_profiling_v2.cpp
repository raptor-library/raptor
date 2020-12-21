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
        printf("Usage: <mat#>\n");
        exit(-1);
    }

    // Grab command line arguments
    int mat = atoi(_argv[1]);

    // Matrix market filenames
    const char* mat1 = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    const char* mat2 = "../../../../../mtx_market_matrices/G3_circuit.pm";
    const char* mat3 = "../../../../../mtx_market_matrices/Hook_1498.pm";

    // Read in matrix
    ParCSRMatrix* A; 
    if (mat == 1)
    {
        A = readParMatrix(mat1);
    }
    else if (mat == 2)
    {
        A = readParMatrix(mat2);
    }
    else if (mat == 3)
    {
        A = readParMatrix(mat3);
    }

    // Declare vectors and residual variables
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    aligned_vector<double> residuals;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Setup for Timings
    double* comp_time = new double[1];
    double* bv_time = new double[1];
    init_profile();
    *comp_time = 0.0;
    
    // Call CG
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    //EKCG(A, x, b, nrhs, residuals, 1e-8, A->global_num_rows, comp_time, bv_time, tap_comm);
    //EKCG(A, x, b, nrhs, residuals, 1e-8, 10, comp_time, bv_time, tap_comm);
    CG(A, x, b, residuals, 1e-8, 100, comp_time);

    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("CG");
    double t;
    MPI_Allreduce(comp_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("Computation Time: %e\n", t/100.0);
    //MPI_Allreduce(bv_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    //if (rank == 0 && t > 0.0) printf("SpMV Time: %e\n", t);
    
    if (rank == 0 && t > 0.0) printf("-------------------------\n");
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete A;
    delete comp_time;
    MPI_Finalize();
    return 0;

} // end of main() //
