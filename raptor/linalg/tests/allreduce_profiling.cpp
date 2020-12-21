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

    // Declare vectors and residual variables
    ParBVector x_par(A->global_num_rows, A->local_num_rows, nrhs);
    BVector b(nrhs, nrhs);
    ParBVector x2_par(A->global_num_rows, A->local_num_rows, nrhs);
    BVector b2(nrhs, nrhs);
    ParBVector x3_par(A->global_num_rows, A->local_num_rows, nrhs);
    BVector b3(nrhs, nrhs);
    aligned_vector<double> temp;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Initial call before init_profile called
    x_par.set_const_value(1.0);
    b.set_const_value(2.0);
    x2_par.set_const_value(1.0);
    b2.set_const_value(2.0);
    x3_par.set_const_value(1.0);
    b3.set_const_value(2.0);

    // Holds values for timings
    double collective_t = 0.0;

    // Perform profiling
    int iterations = 100;
    for (int i =0; i < iterations; i++) 
    {
        x_par.local->mult_T(*(x_par.local), b);

        MPI_Barrier(MPI_COMM_WORLD);

        collective_t -= MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &(b.values[0]), nrhs*nrhs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        collective_t += MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);

        x3_par.local->mult_T(*(x3_par.local), b3);

        /*MPI_Barrier(MPI_COMM_WORLD);

        collective_t -= MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &(b.values[0]), nrhs*nrhs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        collective_t += MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);*/

        x_par.local->mult_T(*(x_par.local), b);
        x2_par.local->mult_T(*(x2_par.local), b2);
        std::copy(b.values.begin(), b.values.end(), back_inserter(temp));
        std::copy(b2.values.begin(), b2.values.end(), back_inserter(temp));
        std::copy(b3.values.begin(), b3.values.end(), back_inserter(temp));

        MPI_Barrier(MPI_COMM_WORLD);

        collective_t -= MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &(temp[0]), 3*nrhs*nrhs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        collective_t += MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);

        temp.clear();
    }

    double t;
    MPI_Reduce(&collective_t, &t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 & t > 0) printf("Collective Comm Time: %e\n", t/iterations);
    
    delete A;
    MPI_Finalize();
    return 0;

} // end of main() //
