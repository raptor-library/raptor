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
    int grid[2] = {5, 5};
    double eps = 0.001;
    double theta = M_PI / 8.0;

    ParCSRMatrix* A;
    ParCSRMatrix* A_3step;
    ParCSRMatrix* A_2step;
    ParCSRMatrix* A_opt;
    ParCSRMatrix* S;
    //ParVector x;

    double* stencil = diffusion_stencil_2d(eps, theta);
    A_3step = par_stencil_grid(stencil, grid, dim);
    A_2step = par_stencil_grid(stencil, grid, dim);
    A_opt = par_stencil_grid(stencil, grid, dim);
    A = par_stencil_grid(stencil, grid, dim);

    //if (rank == 0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);

    /*int grid[3] = {5, 5, 5};
    double* stencil = laplace_stencil_27pt();
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 3);
    if(rank==0) printf("A %d x %d\n", A->global_num_rows, A->global_num_rows);*/

    // SETUP DIFFERENT TAP COMMUNICATORS FOR DIFFERENT RANKS - INIT_TAP_COMMUNICATORS

    /*if (rank % 2)
    {
        // SIMPLE TAP COMM
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
    }
    else
    {
        // 3-step TAP COMM
        A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    }*/

    // 2-step TAP COMM
    A_2step->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false);
    // 3-step TAP COMM
    A_3step->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map);
    // Optimal TAP COMM
    A_opt->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, false, RAPtor_MPI_COMM_WORLD, 1);

    NonContigData* local_R_recv_3step = (NonContigData*) A_3step->tap_comm->local_R_par_comm->recv_data;
    NonContigData* local_R_recv_2step = (NonContigData*) A_2step->tap_comm->local_R_par_comm->recv_data;
    NonContigData* local_R_recv_opt = (NonContigData*) A_opt->tap_comm->local_R_par_comm->recv_data;
    NonContigData* local_R_send_3step = (NonContigData*) A_3step->tap_comm->local_R_par_comm->send_data;
    NonContigData* local_R_send_2step = (NonContigData*) A_2step->tap_comm->local_R_par_comm->send_data;
    NonContigData* local_R_send_opt = (NonContigData*) A_opt->tap_comm->local_R_par_comm->send_data;

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d 3step R recv indices ", rank);
            for (int i = 0; i < local_R_recv_3step->indices.size(); i++)
            {
                printf("%d ", local_R_recv_3step->indices[i]);
            }
            printf("\n");
            printf("%d 3step R send indices ", rank);
            for (int i = 0; i < local_R_send_3step->indices.size(); i++)
            {
                printf("%d ", local_R_send_3step->indices[i]);
            }
            printf("\n");
            fflush(stdout);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    fflush(stdout);
    if (rank == 0) printf("-------------------------------------\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d 2step R recv indices ", rank);
            for (int i = 0; i < local_R_recv_2step->indices.size(); i++)
            {
                printf("%d ", local_R_recv_2step->indices[i]);
            }
            printf("\n");
            printf("%d 2step R send indices ", rank);
            for (int i = 0; i < local_R_send_2step->indices.size(); i++)
            {
                printf("%d ", local_R_send_2step->indices[i]);
            }
            printf("\n");
            fflush(stdout);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    fflush(stdout);
    if (rank == 0) printf("-------------------------------------\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
   
    int msg_cap = 1; 
    // Optimal TAP COMM    
    A->tap_comm = new TAPComm(A->partition, A->off_proc_column_map, A->on_proc_column_map, true, RAPtor_MPI_COMM_WORLD, msg_cap);
    
    // INSERT 2-step COMMUNICATION TEST HERE
    aligned_vector<int> states(A->global_num_rows);
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
    aligned_vector<int>& recvbuf_3step = A_3step->tap_comm->communicate(states);
    aligned_vector<int>& recvbuf_2step = A_2step->tap_comm->communicate(states);

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
    delete A_3step;
    delete A_2step;
    delete A_opt;
    delete[] stencil;

    setenv("PPN", "16", 1);

} // end of TEST(ParAnisoSpMVTest, TestsInUtil) //
