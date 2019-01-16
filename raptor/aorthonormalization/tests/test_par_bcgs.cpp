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
    
    /*for (int i = 0; i < num_procs; i++)
    {
        for (int j = 0; j < P_par->local->b_vecs; j++)
        {
            for (int k = 0; k < P_par->local->num_values; k++)
            {
                printf("P_par[%d][%d] %f\n", j, k, P_par->local->values[j*P_par->local->num_values + k]);
            }
        }
    }*/
    
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
    
    /*printf("---------------- Q1_par ---------------------\n");
    for (int i = 0; i < num_procs; i++)
    {
        for (int j = 0; j < Q1_par->local->b_vecs; j++)
        {
            for (int k = 0; k < Q1_par->local->num_values; k++)
            {
                printf("[%d][%d] %f\n", j, k, Q1_par->local->values[j*Q1_par->local->num_values + k]);
            }
            printf("-----\n");
        }
    }*/

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
    
    /*printf("---------------- Q2_par ---------------------\n");
    for (int i = 0; i < num_procs; i++)
    {
        for (int j = 0; j < Q2_par->local->b_vecs; j++)
        {
            for (int k = 0; k < Q2_par->local->num_values; k++)
            {
                printf("[%d][%d] %f\n", j, k, Q2_par->local->values[j*Q1_par->local->num_values + k]);
            }
            printf("-----\n");
        }
    }*/

    
    //A->mult(*P_par, *P_par);
    BCGS(A, *Q1_par, *Q2_par, *P_par);
    
    /*printf("---------------- P_par ---------------------\n");
    for (int i = 0; i < num_procs; i++)
    {
        for (int j = 0; j < P_par->local->b_vecs; j++)
        {
            for (int k = 0; k < P_par->local->num_values; k++)
            {
                printf("[%d][%d] %f\n", j, k, P_par->local->values[j*P_par->local->num_values + k]);
            }
            printf("-----\n");
        }
    }*/

    // Insert check
    A->mult(*P_par, *T_par);
    P_par->mult_T(*T_par, T);

    //if (rank == 0) T.print();
    // REPLACE THIS WITH A CHECK AGAINST FILE CONTAINING PYTHON RESULT

    CGS(A, *P_par);

    for (int n = 0; n < num_procs; n++)
    {
        if (rank == n)
        {
            printf("---------------- P_par ---------------------\n");
            for (int i = 0; i < num_procs; i++)
            {
                for (int j = 0; j < P_par->local->b_vecs; j++)
                {
                    for (int k = 0; k < P_par->local->num_values; k++)
                    {
                        printf("[%d][%d] %f\n", j, k, P_par->local->values[j*P_par->local->num_values + k]);
                    }
                    printf("-----\n");
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
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
