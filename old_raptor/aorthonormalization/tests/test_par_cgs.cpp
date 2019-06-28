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
    
    int grid[2] = {3, 3};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    double val;
    int W_bvecs = 4;
    double *alphas = new double[W_bvecs];
    
    ParBVector *Q1_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *Q2_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *P_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *T_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    Vector T(W_bvecs, W_bvecs);

    // Set P_par
    P_par->set_const_value(0.0);

    P_par->add_val(4.0, 0, 0);
    P_par->add_val(4.0, 0, 1);
    P_par->add_val(4.0, 0, 2);

    P_par->add_val(3.0, 1, 3);
    P_par->add_val(2.0, 1, 4);
    
    P_par->add_val(3.0, 2, 5);
    P_par->add_val(4.0, 2, 6);

    P_par->add_val(1.0, 3, 7);
    P_par->add_val(2.0, 3, 8);
    
    // Set Q1_par
    Q1_par->set_const_value(0.0);
    
    Q1_par->add_val(3.0, 0, 0);
    Q1_par->add_val(4.0, 0, 1);
    Q1_par->add_val(3.0, 0, 2);

    Q1_par->add_val(3.0, 1, 3);
    Q1_par->add_val(1.0, 1, 4);
    
    Q1_par->add_val(1.0, 2, 5);
    Q1_par->add_val(2.0, 2, 6);

    Q1_par->add_val(1.0, 3, 7);
    Q1_par->add_val(2.0, 3, 8);
    
    // Set Q2_par
    Q2_par->set_const_value(0.0);
    
    Q2_par->add_val(2.0, 0, 0);
    Q2_par->add_val(1.0, 0, 1);
    Q2_par->add_val(2.0, 0, 2);

    Q2_par->add_val(3.0, 1, 3);
    Q2_par->add_val(1.0, 1, 4);
    
    Q2_par->add_val(1.0, 2, 5);
    Q2_par->add_val(2.0, 2, 6);

    Q2_par->add_val(4.0, 3, 7);
    Q2_par->add_val(1.0, 3, 8);
    
    BCGS(A, *Q1_par, *Q2_par, *P_par);
    
    FILE* f = fopen("../../../../test_data/bcgs_soln.txt", "r");
    for (int i = 0; i < P_par->first_local; i++)
    {
        for (int j = 0; j < W_bvecs; j++)
        {
            fscanf(f, "%lg\n", &val);
        }
    }
    for (int i = 0; i < P_par->local_n; i++)
    {
        for (int j = 0; j < W_bvecs; j++)
        {
            fscanf(f, "%lg", &val);
            ASSERT_NEAR(P_par->local->values[j*P_par->local_n + i], val, 1e-06);
        }
    }
    fclose(f);

    CGS(A, *P_par);
    
    f = fopen("../../../../test_data/cgs_soln.txt", "r");
    for (int i = 0; i < P_par->first_local; i++)
    {
        for (int j = 0; j < W_bvecs; j++)
        {
            fscanf(f, "%lg\n", &val);
        }
    }
    for (int i = 0; i < P_par->local_n; i++)
    {
        for (int j = 0; j < W_bvecs; j++)
        {
            fscanf(f, "%lg", &val);
            ASSERT_NEAR(P_par->local->values[j*P_par->local_n + i], val, 1e-06);
        }
    }
    fclose(f);

    delete[] stencil;
    delete A;
    delete Q1_par;
    delete Q2_par;
    delete P_par;
    delete T_par;
    delete alphas;

} // end of TEST(ParBVectorMultTest, TestsInUtil) //
