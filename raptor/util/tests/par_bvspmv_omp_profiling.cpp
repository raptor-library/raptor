// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc < 4)
    {
        printf("Usage: <block_vecs> <test> <timings>\n");
        exit(-1);
    }

    // Grab command line arguments
    int block_vecs = atoi(argv[1]);
    int test = atoi(argv[2]);
    int timings = atoi(argv[4]);

    bool tap;

    // Setup matrix
    //int grid[2] = {2500, 2500};
    /*int grid[2] = {1000, 1000};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);*/

    FILE* f;
    //const char* mfem_fn = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    const char* mfem_fn = "../../../../../mfem_matrices/mfem_grad_div_241.pm";
    ParCSRMatrix* A = readParMatrix(mfem_fn);

    // Setup BVs
    ParBVector x(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, block_vecs);
    ParBVector b(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col, block_vecs);
    
    // Determine communication based on test 
    if (test == 0)
    {
        tap = false; // standard
        A->comm = NULL;
    }
    else if (test == 1)
    {
        tap = true; // 3-step
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    }
    else
    {
        tap = true; // 2-step
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
    }

    // Perform multiple BVSpMVs
    for (int i = 0; i < timings; i++)
    {
        x.set_const_value(1.0);
        A->mult(x, b, tap);
    }
    
    delete A;
    //delete[] stencil;
    
    MPI_Finalize();
    return 0;
} // end of main() //
