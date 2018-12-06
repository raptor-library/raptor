#include <assert.h>

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "aorthonormalization/par_cgs.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;

} // end of main() //

TEST(ParCGSTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int num_vectors = 4;
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    double val;
    int Q_bvecs = 5;
    int W_bvecs = 2;
    int global_n = 100;
    int local_n = global_n / num_procs;
    int first_n = rank * ( global_n / num_procs);

    if (global_n % num_procs > rank)
    {
        local_n++;
        first_n += rank;
    }
    else
    {
        first_n += (global_n % num_procs);
    }
    
    ParBVector *Q1_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *Q2_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *P_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);

    // Test Aortho P against Q_par
    BCGS(A, *Q1_par, *Q2_par, *P_par);

    // Insert check
    
    // Test Aortho of P cols
    QR(A, *P_par);

    // Insert check

    delete[] stencil;
    delete A;
    delete Q1_par;
    delete Q2_par;
    delete P_par;

} // end of TEST(ParBVectorMultTest, TestsInUtil) //
