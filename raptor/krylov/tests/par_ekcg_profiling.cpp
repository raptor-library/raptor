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

    // Matrix market filenames
    const char* mat1 = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    const char* mat2 = "../../../../../mtx_market_matrices/G3_circuit.pm";
    const char* mat3 = "../../../../../mtx_market_matrices/Hook_1498.pm";
    // Residual output filenames
    const char* res_filename1 = "mfem_dg_diffusion_res.txt";
    const char* res_filename2 = "G3_circuit_res.txt";
    const char* res_filename3 = "Hook_1498_res.txt";
    const char* min_res_filename1 = "mfem_dg_diffusion_res_mincomm.txt";
    const char* min_res_filename2 = "G3_circuit_res_mincomm.txt";
    const char* min_res_filename3 = "Hook_1498_res_mincomm.txt";
    // For writing out residual files
    FILE* f;

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
    aligned_vector<double> residuals_mincomm;

    if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    // Setup for Timings
    bool tap_comm = false;
    double* comp_time = new double[1];
    double* bv_time = new double[1];
    init_profile();
    *comp_time = 0.0;
    *bv_time = 0.0;
    
    // Setup correct node aware communication
    if (nap_version == 2)
    {
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
        tap_comm = true;
    }
    else if (nap_version == 3) 
    {
        // tap comm will get created when first tap aware spmv called
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
        tap_comm = true;
    }

    // Call SRECG
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    //EKCG(A, x, b, nrhs, residuals, 1e-8, A->global_num_rows, comp_time, bv_time, tap_comm);
    EKCG(A, x, b, nrhs, residuals, 1e-8, 5, comp_time, bv_time, tap_comm);

    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("EKCG");
    double t;
    MPI_Allreduce(comp_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("Computation Time: %e\n", t/5.0);
    MPI_Allreduce(bv_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("BVSpMV Time: %e\n", t);
    
    // Print out residuals to file
    /*if (rank == 0)
    {
        if (mat == 1) f = fopen(res_filename1, "w");
        if (mat == 2) f = fopen(res_filename2, "w");
        if (mat == 3) f = fopen(res_filename3, "w");
        for (int i = 0; i < residuals.size(); i++)
        {
            fprintf(f, "%e\n", residuals[i]);
        }
        fclose(f);
    }*/

    if (rank == 0 && t > 0.0) printf("-------------------------\n");
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Setup for Timings
    delete comp_time;
    delete bv_time;
    comp_time = new double[1];
    bv_time = new double[1];
    *comp_time = 0.0;
    *bv_time = 0.0;
    
    // Call SRECG
    x.set_const_value(0.0);
    init_profile();
    //EKCG_MinComm(A, x, b, nrhs, residuals_mincomm, 1e-8, A->global_num_rows, comp_time, bv_time, tap_comm);
    EKCG_MinComm(A, x, b, nrhs, residuals_mincomm, 1e-8, 100, comp_time, bv_time, tap_comm);

    // Finalize profile for communication timings
    finalize_profile();

    // Print times
    print_profile("EKCG MinComm");
    MPI_Allreduce(comp_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("Computation Time: %e\n", t/5.0);
    MPI_Allreduce(bv_time, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && t > 0.0) printf("BVSpMV Time: %e\n", t);

    // Print out residuals to file
    /*if (rank == 0)
    {
        if (mat == 1) f = fopen(min_res_filename1, "w");
        if (mat == 2) f = fopen(min_res_filename2, "w");
        if (mat == 3) f = fopen(min_res_filename3, "w");
        for (int i = 0; i < residuals_mincomm.size(); i++)
        {
            fprintf(f, "%e\n", residuals_mincomm[i]);
        }
        fclose(f);
    }*/
    
    delete A;
    delete comp_time;
    delete bv_time;
    MPI_Finalize();
    return 0;

} // end of main() //
