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

TEST(TAPAnisoSpMVTest, TestsInUtil)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    setenv("PPN", "4", 1);
    
    int dim = 2;
    int grid[2] = {10, 10};
    double eps = 0.001;
    double theta = M_PI / 8.0;

    ParCSRMatrix* A;
    ParCSRMatrix* S;
    //ParVector x;

    double* stencil = diffusion_stencil_2d(eps, theta);
    A = par_stencil_grid(stencil, grid, dim);

    // 1. SETUP TAP COMMUNICATORS THE WAY WE DO IN EXTEND HIERARCHY - INIT_TAP_COMMUNICATORS

    // TEST SIMPLE TAP COMM
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
    
    // INSERT 2-step COMMUNICATION TEST HERE
    aligned_vector<int> states(5);
    for (int i = 0; i < states.size(); i++)
    {
        states[i] = rank;
    }    
    
    aligned_vector<int>& recvbuf = A->comm->communicate(states);

    

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("before communicate\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    aligned_vector<int>& recvbuf2 = A->tap_comm->communicate(states);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("after communicate\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    /*MPI_Status recv_status;
    int count;
    MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recv_status);
    MPI_Get_count(&recv_status, MPI_INT, &count);
    //MPI_Barrier(MPI_COMM_WORLD);
    printf("%d flag %d\n", rank, count);*/
    
    fflush(stdout);
    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d recvbuf ", rank);
            for (int i = 0; i < recvbuf.size(); i++)
            {
                printf("%d ", recvbuf[i]);
            }
            printf("\n");
            printf("%d recvbuf2 ", rank);
            for (int i = 0; i < recvbuf2.size(); i++)
            {
                printf("%d ", recvbuf2[i]);
            }
            printf("\n");
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    delete A;
    delete[] stencil;

    setenv("PPN", "16", 1);

} // end of TEST(ParAnisoSpMVTest, TestsInUtil) //
