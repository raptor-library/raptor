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
    //int grid[2] = {50, 50};
    int grid[2] = {2, 2};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    for (int i = 0; i < A->global_num_rows; i++)
    {
        int start = A->on_proc->idx1[i];
        int stop = A->on_proc->idx1[i+1];
        for (int j = start; j < stop; j++)
        {
            A->on_proc->vals[j] = 1.0;
        }
    }
    //A->on_proc->print();

    double val;
    //int Q_bvecs = 5;
    int W_bvecs = 3;
    double *alphas = new double[W_bvecs];
    /*int global_n = 100;
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
    }*/
    
    ParBVector *Q1_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *Q2_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *P_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    ParBVector *T_par = new ParBVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row, W_bvecs);
    Vector t(W_bvecs);

    // Test CGS
    for (int i = 0; i < W_bvecs; i++)
    {
        alphas[i] = i+1;
    }
    P_par->set_const_value(1.0);
    P_par->scale(1, alphas);
    for (int i = 0; i < W_bvecs; i++)
    {
        P_par->local->values[i*P_par->local_n + i] = i+2;
        if (i < W_bvecs - 1) P_par->local->values[i*P_par->local_n + i + 1] = i+3;
    }

    CGS(A, *P_par);
    CGS(A, *P_par);
    
    // Check for Aortho of P_par columns
    A->mult(*P_par, *T_par);
    P_par->mult_T(*T_par, t);
    for (int i = 0; i < W_bvecs; i++)
    {
        printf("t[%d] %lg\n", i, t[i]);
        //ASSERT_NEAR( t.values[i], 1.0, 1e-06);
    }

    // Test MGS
    P_par->set_const_value(0.0); 
    for (int i = 0; i < W_bvecs; i++)
    {
        P_par->local->values[i*P_par->local_n + i] = i+2;
        if (i < W_bvecs - 1) P_par->local->values[i*P_par->local_n + i + 1] = i+3;
    }
    //MGS(A, *P_par);

    // Check for Aortho of P_par columns
    A->mult(*P_par, *T_par);
    P_par->mult_T(*T_par, t);
    for (int i = 0; i < W_bvecs; i++)
    {
        printf("t[%d] %lg\n", i, t[i]);
        //ASSERT_NEAR( t.values[i], 1.0, 1e-06);
    }

    // Test Aortho P against Q_par
    Vector T(W_bvecs, W_bvecs);
    A->mult(*P_par, *P_par);
    BCGS(A, *Q1_par, *Q2_par, *P_par);

    // Insert check
    A->mult(*P_par, *T_par);
    P_par->mult_T(*T_par, T);

    if (rank == 0) T.print();   

    delete[] stencil;
    delete A;
    delete Q1_par;
    delete Q2_par;
    delete P_par;
    delete T_par;
    delete alphas;

} // end of TEST(ParBVectorMultTest, TestsInUtil) //
