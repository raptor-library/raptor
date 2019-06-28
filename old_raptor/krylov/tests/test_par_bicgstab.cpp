// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_bicgstab.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;

// *******************************************************************
//              BiCGStab not working?
// *******************************************************************

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParBiCGStabTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    bool compare_res = false;
    bool check_soln = true;

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    ParVector* x = new ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector* b = new ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals;

    x->set_const_value(1.0);
    A->mult(*x, *b);
    x->set_const_value(0.0);

    BiCGStab(A, *x, *b, residuals);

    // Just testing the first 10 residuals
    if(rank == 0 && compare_res){
        FILE* f = fopen("../../../../test_data/bicgstab_res.txt", "r");
        double res;
        for (int i = 0; i < 10; i++)
        {
            fscanf(f, "%lf\n", &res);
            ASSERT_NEAR(res, residuals[i], 1e-06);
        }
        fclose(f);
    }
    
    if (check_soln)
    {
        for (int i = 0; i < x->local_n; i++)
        {
            ASSERT_NEAR(x->local->values[i], 1.0, 1e-03);
        }        
    }

    delete[] stencil;
    delete A;
    delete x;
    delete b;
    
} // end of TEST(ParBiCGStabTest, TestsInKrylov) //
