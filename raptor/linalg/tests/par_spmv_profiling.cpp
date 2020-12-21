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
        printf("Usage: <mat#> <nap_version>\n");
        exit(-1);
    }

    // Grab command line arguments
    int mat = atoi(_argv[1]);
    int nap_version = atoi(_argv[2]);

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

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Setup for Timings
    bool tap_comm = false;
    
    // Setup correct node aware communication
    if (nap_version == 2)
    {
        if (rank == 0) printf("2-step Comm\n");
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
        tap_comm = true;
    }
    else if (nap_version == 3) 
    {
        if (rank == 0) printf("3-step Comm\n");
        // tap comm will get created when first tap aware spmv called
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
        tap_comm = true;
    }
    else
    {
        if (rank == 0 ) printf("Standard Comm\n");
    }

    // Initial call before init_profile called
    x.set_const_value(1.0);
    A->mult(x, b);

    // Perform profiling
    init_profile();
    for (int i =0; i < 5; i++) A->mult(x, b);
    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("SpMV");
    
    delete A;
    MPI_Finalize();
    return 0;

} // end of main() //
