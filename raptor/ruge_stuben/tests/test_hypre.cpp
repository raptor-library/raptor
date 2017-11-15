// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "gtest/gtest.h"

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "tests/hypre_compare.hpp"
#include "gallery/external/hypre_wrapper.hpp"
#include "multilevel/par_multilevel.hpp"
#include "multilevel/multilevel.hpp"

using namespace raptor;

void form_hypre_weights(std::vector<double>& weights, int n_rows)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (n_rows)
    {
        weights.resize(n_rows);
        int seed = 2747 + rank;
        int a = 16807;
        int m = 2147483647;
        int q = 127773;
        int r = 2836;
        for (int i = 0; i < n_rows; i++)
        {
            int high = seed / q;
            int low = seed % q;
            int test = a * low - r * high;
            if (test > 0) seed = test;
            else seed = test + m;
            weights[i] = ((double)(seed) / m);
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(TestHypre, TestsInRuge_Stuben)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0;
    double hypre_setup, hypre_solve;
    double raptor_setup, raptor_solve;

    int coarsen_type = 0; // CLJP
    //int coarsen_type = 6; // FALGOUT
    //int interp_type = 3; // Direct Interp
    int interp_type = 0; // Classical Mod Interp
    double strong_threshold = 0.25;
    int agg_num_levels = 0;
    int p_max_elmts = 0;

    int cache_len = 10000;
    int num_tests = 2;

    std::vector<double> cache_array(cache_len);

    int dim = 2;
    int grid[2] = {100, 100};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    A = par_stencil_grid(stencil, grid, dim);
    delete[] stencil;
    x = ParVector(A->global_num_cols, A->on_proc_num_cols, A->partition->first_local_col);
    b = ParVector(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    x.set_const_value(1.0);
    A->mult(x, b);

    // Convert system to Hypre format 
    HYPRE_IJMatrix A_h_ij = convert(A);
    HYPRE_IJVector x_h_ij = convert(&x);
    HYPRE_IJVector b_h_ij = convert(&b);
    hypre_ParCSRMatrix* A_h;
    HYPRE_IJMatrixGetObject(A_h_ij, (void**) &A_h);
    hypre_ParVector* x_h;
    HYPRE_IJVectorGetObject(x_h_ij, (void **) &x_h);
    hypre_ParVector* b_h;
    HYPRE_IJVectorGetObject(b_h_ij, (void **) &b_h);

    HYPRE_Solver solver_data = hypre_create_hierarchy(A_h, x_h, b_h, 
                                coarsen_type, interp_type, p_max_elmts, agg_num_levels, 
                                strong_threshold);
    ParMultilevel* ml = new ParMultilevel(A, strong_threshold);

    HYPRE_Int num_levels = hypre_ParAMGDataNumLevels((hypre_ParAMGData*) solver_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray((hypre_ParAMGData*) solver_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray((hypre_ParAMGData*) solver_data);

    assert(num_levels == ml->num_levels);
    for (int i = 0; i < num_levels; i++)
    {
        compare(ml->levels[i]->A, A_array[i]);
    }
    for (int i = 0; i < num_levels - 1; i++)
    {
        compare(ml->levels[i]->P, P_array[i]);
    }

    hypre_BoomerAMGDestroy(solver_data); 
    delete ml;

    delete A;
    HYPRE_IJMatrixDestroy(A_h_ij);
    HYPRE_IJVectorDestroy(x_h_ij);
    HYPRE_IJVectorDestroy(b_h_ij);
} // end of TEST(TestHypre, TestsInRuge_Stuben)

