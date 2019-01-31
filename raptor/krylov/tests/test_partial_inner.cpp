// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParPartialInnerTest, TestsInKrylov)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm inner_comm, roots_comm; 

    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(0.001, M_PI/8.0);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    ParVector x(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);
    ParVector y(A->global_num_rows, A->local_num_rows, A->partition->first_local_row);

    x.set_const_value(1.0);
    y.set_const_value(1.0);

    data_t inner;
    int color, root_color, root, procs_in_group, part_global, groups;
    std::vector<double> fracs = {0.5, 0.333, 0.25, 0.2, 0.1};

    for (int i=0; i<fracs.size(); i++) {
        inner_comm = MPI_COMM_NULL;
        roots_comm = MPI_COMM_NULL;
        create_partial_inner_comm(inner_comm, roots_comm, fracs[i], x, color, root_color, root, procs_in_group, part_global);
       
        groups = 1 / fracs[i];
        for (int j=0; j<groups; j++) {
            inner = partial_inner(inner_comm, roots_comm, x, y, color, j, root, procs_in_group, part_global);
            ASSERT_NEAR(inner, x.global_n, 1e-01);
            MPI_Barrier(MPI_COMM_WORLD); 
        }
    }

    MPI_Comm_free(&inner_comm);
    MPI_Comm_free(&roots_comm);
    delete[] stencil;
    delete A;
    
} // end of TEST(ParPartialInnerTest, TestsInKrylov) //

