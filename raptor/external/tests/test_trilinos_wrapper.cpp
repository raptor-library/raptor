// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"
#include "external/trilinos_wrapper.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //


TEST(TrilinosWrapperTest, TestsInExternal)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create RAPtor Matrix to be solved
    int n = 100;
    int dim = 2;
    std::vector<int> grid;
    grid.resize(dim, n);
    double eps = 0.001;
    double theta = M_PI/4.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid.data(), dim);
    ParVector x(A->global_num_rows, A->local_num_rows);
    delete[] stencil;
    x.set_rand_values();
    ParVector b(A->global_num_rows, A->local_num_rows);
    b.set_const_value(0.0);

    Epetra_CrsMatrix* A_epetra = epetra_convert(A);
    Epetra_Vector* x_epetra = epetra_convert(x);
    Epetra_Vector* b_epetra = epetra_convert(b);

    std::vector<double> x_data(A->local_num_rows);
    std::vector<double> b_data(A->local_num_rows);
    x_epetra->ExtractCopy(x_data.data());

    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_EQ(x[i], x_data[i]);
    }

    A->mult(x, b);
    A_epetra->Multiply(false, *x_epetra, *b_epetra);
    b_epetra->ExtractCopy(b_data.data());

    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b[i], b_data[i], 1e-06);
    }

    std::vector<int> coords(A->local_num_rows * dim);
    int global_row;
    // global row i is at coord ((i / n) , (i % n))
    for (int i = 0; i < A->local_num_rows; i++)
    {
        global_row = A->local_row_map[i];
        coords[dim*i] = global_row / n;
        coords[dim*i+1] = global_row % n;
    }
    AztecOO* ml_solver = create_ml_hierarchy(A_epetra, x_epetra, b_epetra, dim, coords.data());
    delete ml_solver;

    // Delete Matrix
    delete A_epetra;
    delete x_epetra;
    delete b_epetra;
    delete A;
}
