// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp = RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(ParBVectorAnisoTAPSpMVTest, TestsInUtil)
{
    setenv("PPN", "4", 1);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int vecs_in_block = 3;

    ParCSRMatrix* P;

    for (int i = 0; i < P->on_proc->vals.size(); i++)
    {
        P->on_proc->vals[i] = 1.0;
    }
    for (int i = 0; i < P->off_proc->vals.size(); i++)
    {
        P->off_proc->vals[i] = 1.0;
    }

    aligned_vector<int> on_proc_idx1;
    aligned_vector<int> on_proc_idx2;
    aligned_vector<int> off_proc_idx1;
    aligned_vector<int> off_proc_idx2;
    if (rank == 0)
    {
       aligned_vector<int> on_proc_idx1 = {0, 0, 0, 0};
       aligned_vector<int> on_proc_idx2 = {};
       aligned_vector<int> off_proc_idx1 = {0, 2, 3};
       aligned_vector<int> off_proc_idx2 = {0, 1, 1};
    }
    else if (rank == 1)
    {
       aligned_vector<int> on_proc_idx1 = {0, 0, 1, 1};
       aligned_vector<int> on_proc_idx2 = {0};
       aligned_vector<int> off_proc_idx1 = {0, 1, 2, 3};
       aligned_vector<int> off_proc_idx2 = {0, 1, 2};
    }
    else if (rank == 2)
    {
       aligned_vector<int> on_proc_idx1 = {0, 1, 2, 2};
       aligned_vector<int> on_proc_idx2 = {0, 0};
       aligned_vector<int> off_proc_idx1 = {0, 1, 2, 3};
       aligned_vector<int> off_proc_idx2 = {2, 1, 0};
    }
    else if (rank == 3)
    {
       aligned_vector<int> on_proc_idx1 = {0, 0, 0, 0};
       aligned_vector<int> on_proc_idx2 = {};
       aligned_vector<int> off_proc_idx1 = {0, 1};
       aligned_vector<int> off_proc_idx2 = {0};
    }
    else if (rank == 4)
    {
       aligned_vector<int> on_proc_idx1 = {0, 1, 1};
       aligned_vector<int> on_proc_idx2 = {0};
       aligned_vector<int> off_proc_idx1 = {0, 1, 2};
       aligned_vector<int> off_proc_idx2 = {1, 0};
    }
    else if (rank == 5)
    {
       aligned_vector<int> on_proc_idx1 = {0, 1};
       aligned_vector<int> on_proc_idx2 = {0};
       aligned_vector<int> off_proc_idx1 = {0, 1};
       aligned_vector<int> off_proc_idx2 = {0};
    }
    else if (rank == 6)
    {
       aligned_vector<int> on_proc_idx1 = {0, 1, 2};
       aligned_vector<int> on_proc_idx2 = {0, 0};
       aligned_vector<int> off_proc_idx1 = {0, 1, 2};
       aligned_vector<int> off_proc_idx2 = {0, 1};
    }
    else if (rank == 7)
    {
       aligned_vector<int> on_proc_idx1 = {0, 0, 1};
       aligned_vector<int> on_proc_idx2 = {0};
       aligned_vector<int> off_proc_idx1 = {0, 2, 4};
       aligned_vector<int> off_proc_idx2 = {1, 2, 0, 2};
    }

    setenv("PPN", "16", 1);

} // end of TEST(ParBVectorAnisoTAPSpMVTest, TestsInUtil) //
