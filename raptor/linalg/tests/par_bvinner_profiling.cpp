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
        printf("Usage: <mat#> <nrhs>\n");
        exit(-1);
    }

    // Grab command line arguments
    int mat = atoi(_argv[1]);
    int nrhs = atoi(_argv[2]);

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

    // Declare vectors and array for inner products
    ParBVector x(A->global_num_rows, A->local_num_rows, nrhs);
    double *inner_prods = new double[nrhs];
    double temp;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Initial call before init_profile called
    x.set_const_value(1.0);
    temp = x.inner_product(x, inner_prods);

    // Perform profiling
    int iterations = 100;
    init_profile();
    for (int i =0; i < iterations; i++) temp = x.inner_product(x, inner_prods);
    // Finalize profile for communication timings
    finalize_profile();
    average_profile(iterations);

    // Print times
    print_profile("BVInner");
    
    delete A;
    delete inner_prods;
    MPI_Finalize();
    return 0;

} // end of main() //
