// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "multilevel/multilevel.hpp"
#include "ruge_stuben/ruge_stuben_solver.hpp"
#include "aggregation/smoothed_aggregation_solver.hpp"
#include "gallery/matrix_IO.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/stencil.hpp"

using namespace raptor;

int argc;
char **argv;

int main(int _argc, char** _argv)
{
    ::testing::InitGoogleTest(&_argc, _argv);
    argc = _argc;
    argv = _argv;
    return RUN_ALL_TESTS();

} // end of main() //

TEST(AMGTest, TestsInMultilevel)
{
    int dim = 3;
    
    int grid[3] = {5, 5, 5};

    Multilevel* ml;
    CSRMatrix* A;
    Vector x;
    Vector b;

    double strong_threshold = 0.0;
    aligned_vector<double> res;
    int iter;
    

    double* stencil = laplace_stencil_27pt();
    A = stencil_grid(stencil, grid, dim);
    delete[] stencil;

    x.resize(A->n_rows);
    b.resize(A->n_rows);
    
    printf("Ruge Stuben Solver:\n");
    ml = new RugeStubenSolver(strong_threshold);
    ml->setup(A);

    printf("Num Levels = %d\n", ml->num_levels);
	printf("A\tNRow\tNCol\tNNZ\n");
    for (int i = 0; i < ml->num_levels; i++)
    {
        CSRMatrix* Al = ml->levels[i]->A;
        printf("%d\t%d\t%d\t%d\n", i, Al->n_rows, Al->n_cols, Al->nnz);
    }
	printf("\nP\tNRow\tNCol\tNNZ\n");
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        CSRMatrix* Pl = ml->levels[i]->P;
        printf("%d\t%d\t%d\t%d\n", i, Pl->n_rows, Pl->n_cols, Pl->nnz);
    }
    
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);
    printf("\nSolve Phase Relative Residuals:\n");
    res = ml->get_residuals();
    for (int i = 0; i < iter; i++)
    {
        printf("Res[%d] = %e\n", i, res[i]);
    }
    delete ml;


    printf("\nSmoothed Aggregation Solver:\n");
    ml = new SmoothedAggregationSolver(strong_threshold);
    ml->setup(A);

    printf("Num Levels = %d\n", ml->num_levels);
	printf("A\tNRow\tNCol\tNNZ\n");
    for (int i = 0; i < ml->num_levels; i++)
    {
        CSRMatrix* Al = ml->levels[i]->A;
        printf("%d\t%d\t%d\t%d\n", i, Al->n_rows, Al->n_cols, Al->nnz);
    }
	printf("\nP\tNRow\tNCol\tNNZ\n");
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        CSRMatrix* Pl = ml->levels[i]->P;
        printf("%d\t%d\t%d\t%d\n", i, Pl->n_rows, Pl->n_cols, Pl->nnz);
    }
    
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    iter = ml->solve(x, b);
    printf("\nSolve Phase Relative Residuals:\n");
    res = ml->get_residuals();
    for (int i = 0; i < iter; i++)
    {
        printf("Res[%d] = %e\n", i, res[i]);
    }
    delete ml;


    delete A;

} // end of TEST(AMGTest, TestsInMultilevel) //
