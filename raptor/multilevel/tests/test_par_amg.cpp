// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "multilevel/par_multilevel.hpp"
#include "gallery/par_matrix_IO.hpp"
#include "gallery/diffusion.hpp"
#include "gallery/laplacian27pt.hpp"
#include "gallery/par_stencil.hpp"

using namespace raptor;


int argc;
char **argv;

int main(int _argc, char** _argv)
{
    MPI_Init(&_argc, &_argv);
    
    ::testing::InitGoogleTest(&_argc, _argv);
    argc = _argc;
    argv = _argv;
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(ParAMGTest, TestsInMultilevel)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    int dim = 3;
    int system = 0;
    int grid[3] = {5, 5, 5};

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double strong_threshold = 0.25;

    int cache_len = 10000;
    double* cache_array = new double[cache_len];
    int num_tests = 10;

    double* stencil = laplace_stencil_27pt();
    A = par_stencil_grid(stencil, grid, dim);
    delete[] stencil;

    x.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    b.resize(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    x.set_const_value(1.0);
    A->mult(x, b);
    x.set_const_value(0.0);
    
    ml = new ParMultilevel(A, strong_threshold, CLJP, Classical, SOR, 1, 1.0, 50, -1);

    if (rank == 0)
    {
        printf("Num Levels = %d\n", ml->num_levels);
	    printf("A\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels; i++)
    {
        ParCSRMatrix* Al = ml->levels[i]->A;
	    long lcl_nnz = Al->local_nnz;
	    long nnz;
	    MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	    if (rank == 0)
	    {
            printf("%d\t%d\t%d\t%lu\n", i, Al->global_num_rows, Al->global_num_cols, nnz);
        }
    }

    if (rank == 0)
    {
	printf("\nP\tNRow\tNCol\tNNZ\n");
    }
    for (int i = 0; i < ml->num_levels-1; i++)
    {
        ParCSRMatrix* Pl = ml->levels[i]->P;
	    long lcl_nnz = Pl->local_nnz;
	    long nnz;
	    MPI_Reduce(&lcl_nnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	    if (rank == 0)
	    {
            printf("%d\t%d\t%d\t%lu\n", i, Pl->global_num_rows, Pl->global_num_cols, nnz);
	    }
    }
    
    if (rank == 0)
    {
        printf("\nSolve Phase Relative Residuals:\n");
    }
    ml->solve(x, b);

    delete ml;
    delete A;

} // end of TEST(ParAMGTest, TestsInMultilevel) //
