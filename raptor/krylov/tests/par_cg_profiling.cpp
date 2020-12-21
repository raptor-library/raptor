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
    const char* mat1 = "../../../../../mtx_market_matrices/Bump_2911.mtx";
    const char* mat2 = "../../../../../mtx_market_matrices/G3_circuit.mtx";
    const char* mat3 = "../../../../../mtx_market_matrices/Hook_1498.mtx";
    // Residual output filenames
    const char* res_filename1 = "Bump_2911_res_cg.txt";
    const char* res_filename2 = "G3_circuit_res_cg.txt";
    const char* res_filename3 = "Hook_1498_res_cg.txt";
    // For writing out residual files
    FILE* f;

    // Read in matrix
    ParCSRMatrix* A; 
    if (mat == 1)
    {
        A = read_par_mm(mat1);
    }
    else if (mat == 2)
    {
        A = read_par_mm(mat2);
    }
    else if (mat == 3)
    {
        A = read_par_mm(mat3);
    }

    ParVector x(A->global_num_rows, A->local_num_rows);
    ParVector b(A->global_num_rows, A->local_num_rows);
    aligned_vector<double> residuals;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Setup for Timings
    bool tap_comm = false;
    double* comp_time = new double[1];
    init_profile();
    *comp_time = 0.0;
    
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

    // Call CG
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    //CG(A, x, b, residuals, 1e-5, A->global_num_rows, comp_time, tap_comm);
    CG(A, x, b, residuals, 1e-8, 10, comp_time, tap_comm);

    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("CG");
    double t;
    MPI_Allreduce(comp_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("Computation Time: %e\n", t);
    
    // Print out residuals to file
    if (rank == 0)
    {
        if (mat == 1) f = fopen(res_filename1, "w");
        if (mat == 2) f = fopen(res_filename2, "w");
        if (mat == 3) f = fopen(res_filename3, "w");
        for (int i = 0; i < residuals.size(); i++)
        {
            fprintf(f, "%e\n", residuals[i]);
        }
        fclose(f);
    }
    
    delete A;
    delete comp_time;
    MPI_Finalize();
    return 0;

} // end of main() //
