// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
#include "core/comm_pkg.hpp"
#include "gallery/stencil.hpp"
#include "gallery/par_stencil.hpp"
#include "gallery/diffusion.hpp"

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

    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);

    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    Vector& x_lcl = x.local;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        x_lcl[i] = A->local_row_map[i];
    }

    TAPComm* tap_comm = new TAPComm(A->partition, A->off_proc_column_map);
    std::vector<double>& tap_recv = tap_comm->communicate(x, MPI_COMM_WORLD);
    std::vector<double>& par_recv = A->comm->communicate(x, MPI_COMM_WORLD);
    ASSERT_EQ(tap_recv.size(), par_recv.size());

    for (int i = 0; i < (int) par_recv.size(); i++)
    {
        ASSERT_NEAR(par_recv[i], tap_recv[i], zero_tol);
    }

    delete[] stencil;
    delete A;
    delete tap_comm;


} // end of TEST(TAPCommTest, TestsInCore) //
