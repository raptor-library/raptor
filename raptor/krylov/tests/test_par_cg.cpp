// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "krylov/par_cg.hpp"
#include "multilevel/par_multilevel.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/par_matrix_IO.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParCGTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    bool compare_res = false;
    bool check_soln = false;
    bool print_res_tofile = true;

    //int grid[2] = {50, 50};
    /*int grid[2] = {1000, 1000};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);*/

    FILE* f;
    const char* mfem_fn = "../../../../../mfem_matrices/mfem_dg_diffusion_331.pm";
    ParCSRMatrix* A = readParMatrix(mfem_fn);

    ParMultilevel *ml;
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector b(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    aligned_vector<double> residuals;
    aligned_vector<double> pre_residuals;

    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);

    double b_norm = b.norm(2);
    CG(A, x, b, residuals);
    
    if (rank == 0)
    {
        if (print_res_tofile)
        {
            FILE* f = fopen("cg_raptor_res.txt", "w");
            for (int i = 0; i < residuals.size(); i++)
            {
                fprintf(f, "%lg\n", residuals[i]);
            }
            fclose(f);
        }

	if (compare_res)
	{
	    FILE* f = fopen("../../../../test_data/cg_res.txt", "r");
	    double res;
	    for (int i = 0; i < residuals.size(); i++)
	    {
	        fscanf(f, "%lf\n", &res);
		ASSERT_NEAR(res, residuals[i] * b_norm, 1e-06);
	    }
	    fclose(f);
	}
    }

    if (check_soln)
    {
        for (int i = 0; i < x.local_n; i++)
        {
            ASSERT_NEAR(x.local->values[i], 1.0, 1e-03);
        }        
    }

    // Setup AMG hierarchy
    /*ml = new ParSmoothedAggregationSolver(0.0);
    ml->max_levels = 3;
    ml->setup(A);

    // AMG Preconditioned BiCGStab
    x.set_const_value(0.0);
    PCG(A, ml, x, b, pre_residuals);
    
    if (check_soln)
    {
        for (int i = 0; i < x.local_n; i++)
        {
            ASSERT_NEAR(x.local->values[i], 1.0, 1e-02);
        }        
    }*/

    //delete[] stencil;
    delete A;
    delete ml;
    
} // end of TEST(ParCGTest, TestsInKrylov) //



