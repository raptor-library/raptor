// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
#include "petscmat.h"
#include "external/petsc_wrapper.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, NULL, NULL);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    PetscFinalize();
    MPI_Finalize();
    return temp;
} // end of main() //



TEST(PetscWrapperTest, TestsInExternal)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create RAPtor Matrix to be solved
    int n = 15;
    int dim = 2;
    aligned_vector<int> grid;
    grid.resize(dim, n);
    double eps = 0.001;
    double theta = M_PI/4.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid.data(), dim);
    delete[] stencil;

    PetscErrorCode ierr = 0;
    Mat pmat;
    Vec rhs, sol;
    KSP ksp;
    PC pc;

    // Setup RAPtor Hierarchy
    ierr = petsc_create_preconditioner(A, &ksp, &pmat, &rhs, &sol);

    // Solve RAPtor Hierarchy
    KSPSolve(ksp, rhs, sol);

    // Check if residual less than 1e-05
    double rnorm;
    KSPGetResidualNorm(ksp, &rnorm);
    ASSERT_LE(rnorm, 1e-05); 

    // Detroy RAPtor Preconditioner
    VecDestroy(&rhs);
    VecDestroy(&sol);
    MatDestroy(&pmat);
    KSPDestroy(&ksp);

    // Delete Matrix
    delete A;

    ASSERT_EQ(ierr, 0);

}
 
// end of TEST(PetscWrapperTest, TestsInExternal) //
