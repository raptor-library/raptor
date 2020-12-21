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

    // Grab command line arguments
    int nrhs = 25;

    int max_iter = 2500;

    // Matrix market filenames
    const char* mat = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    // Residual output filenames
    const char* res_filename = "mfem_25.txt";
    // For writing out residual files
    FILE* f;

    // Read in matrix
    ParCSRMatrix* A; 
    A = readParMatrix(mat);

    // Declare vectors and residual variables
    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    aligned_vector<double> residuals;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Call ECG
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    EKCG(A, x, b, nrhs, residuals, 1e-3, max_iter);

    // Print out residuals to file
    if (rank == 0)
    {
        f = fopen(res_filename, "w");
        for (int i = 0; i < residuals.size(); i++)
        {
            fprintf(f, "%e\n", residuals[i]);
        }
        fclose(f);
    }

    delete A;
    MPI_Finalize();
    return 0;

} // end of main() //
