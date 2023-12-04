// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor/raptor.hpp"
#include "raptor/tests/compare.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;

} // end of main() //
TEST(TAPCommTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {25, 25};
    double* stencil = diffusion_stencil_2d(eps, theta);
    std::vector<double> tap_recv;
    std::vector<double> par_recv;
    std::vector<double> tap_simp_recv;

    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    //A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map);
    //TAPComm* simple_tap = new TAPComm(A->partition, A->off_proc_column_map, false);
    A->init_tap_communicators(MPI_COMM_WORLD);

    ParVector x(A->global_num_rows, A->local_num_rows);
    ParCSRMatrix* B = A->copy();

    for (int i = 0; i < A->local_num_rows; i++)
    {
        x[i] = A->local_row_map[i];
    }
    tap_recv = A->tap_comm->communicate(x);
    tap_simp_recv = A->tap_mat_comm->communicate(x);
    par_recv = A->comm->communicate(x);
    ASSERT_EQ(tap_recv.size(), par_recv.size());
    ASSERT_EQ(tap_recv.size(), tap_simp_recv.size());
    for (int i = 0; i < (int)par_recv.size(); i++)
    {
        ASSERT_NEAR(par_recv[i], tap_recv[i], zero_tol);
        ASSERT_NEAR(tap_recv[i], tap_simp_recv[i], zero_tol);
    }

    x.set_rand_values();
    tap_recv = A->tap_comm->communicate(x);
    tap_simp_recv = A->tap_mat_comm->communicate(x);
    par_recv = A->comm->communicate(x);
    ASSERT_EQ(tap_recv.size(), par_recv.size());
    ASSERT_EQ(tap_simp_recv.size(), tap_recv.size());
    for (int i = 0; i < (int)par_recv.size(); i++)
    {
        ASSERT_NEAR(par_recv[i], tap_recv[i], zero_tol);
        ASSERT_NEAR(tap_recv[i], tap_simp_recv[i], zero_tol);
    }

    CSRMatrix* recv_mat = A->comm->communicate(B);
    CSRMatrix* tap_recv_mat = A->tap_comm->communicate(B);
    CSRMatrix* tap_recv_simp_mat = A->tap_mat_comm->communicate(B);
    compare(recv_mat, tap_recv_mat);
    compare(tap_recv_mat, tap_recv_simp_mat);
    delete recv_mat;
    delete tap_recv_mat;
    delete tap_recv_simp_mat;

    delete[] stencil;
    delete A;


} // end of TEST(TAPCommTest, TestsInCore) //
