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

    if (_argc < 3)
    {
        printf("Usage: <mat#> <nrhs> <nap_version>\n");
        exit(-1);
    }

    // Grab command line arguments
    int mat = atoi(_argv[1]);
    int nrhs = atoi(_argv[2]);
    int nap_version = atoi(_argv[3]);
    int msg_cap;
    if (nap_version == 4) msg_cap = atoi(_argv[4]);

    // Matrix market filenames
    const char* mat1 = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    const char* mat2 = "../../../../../mtx_market_matrices/G3_circuit.pm";
    const char* mat3 = "../../../../../mtx_market_matrices/Hook_1498.pm";
    const char* mat4 = "../../../../../mtx_market_matrices/audikw_1.pm";
    const char* mat5 = "../../../../../mtx_market_matrices/bone010.pm";
    const char* mat6 = "../../../../../mtx_market_matrices/Emilia_923.pm";
    const char* mat7 = "../../../../../mtx_market_matrices/Flan_1565.pm";
    const char* mat8 = "../../../../../mtx_market_matrices/Geo_1438.pm";
    const char* mat9 = "../../../../../mtx_market_matrices/ldoor.pm";
    const char* mat10 = "../../../../../mtx_market_matrices/Serena.pm";
    const char* mat11 = "../../../../../mtx_market_matrices/StocF-1465.pm";
    const char* mat12 = "../../../../../mtx_market_matrices/thermal2.pm";

    // Read in matrix
    ParCSRMatrix* A; 
    if (mat == 1) A = readParMatrix(mat1);
    else if (mat == 2) A = readParMatrix(mat2);
    else if (mat == 3) A = readParMatrix(mat3);
    else if (mat == 4) A = readParMatrix(mat4);
    else if (mat == 5) A = readParMatrix(mat5);
    else if (mat == 6) A = readParMatrix(mat6);
    else if (mat == 7) A = readParMatrix(mat7);
    else if (mat == 8) A = readParMatrix(mat8);
    else if (mat == 9) A = readParMatrix(mat9);
    else if (mat == 10) A = readParMatrix(mat10);
    else if (mat == 11) A = readParMatrix(mat11);
    else if (mat == 12) A = readParMatrix(mat12);

    // Declare vectors and residual variables
    ParBVector x(A->global_num_rows, A->local_num_rows, nrhs);
    ParBVector b(A->global_num_rows, A->local_num_rows, nrhs);

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Setup
    // Factor on_proc portion of A into L and U
    CSRMatrix* L = new CSRMatrix(A->on_proc->n_rows, A->on_proc->n_cols);
    CSRMatrix* U = new CSRMatrix(A->on_proc->n_rows, A->on_proc->n_cols);
    A->on_proc->gaussian_elimination(L, U);
    
    // Perform forward and backward substitution with L and U
    
    // Perform profiling of forward and backward substitution multiple times
 
    // Initial call before init_profile called
    x.set_const_value(1.0);
    A->mult(x, b);

    // Perform profiling
    init_profile();
    for (int i =0; i < 100; i++) A->mult(x, b);
    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("BVSpMV");
    
    delete A;
    MPI_Finalize();
    return 0;

} // end of main() //
