// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"

#include "raptor.hpp"

using namespace raptor;

void compare(ParComm* standard_comm, ParComm* new_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int proc, idx;
    int start, end;
    int standard_start, standard_end;

    // Compare ParComms
    aligned_vector<int> proc_to_idx(num_procs);
    for (int i = 0; i < standard_comm->send_data->num_msgs; i++)
    {
        proc = standard_comm->send_data->procs[i];
        proc_to_idx[proc] = i;
    }
    ASSERT_EQ(standard_comm->send_data->num_msgs, new_comm->send_data->num_msgs);
    ASSERT_EQ(standard_comm->send_data->size_msgs, new_comm->send_data->size_msgs);
    for (int i = 0; i < new_comm->send_data->num_msgs; i++)
    {
        proc = new_comm->send_data->procs[i];
        start = new_comm->send_data->indptr[i];
        end = new_comm->send_data->indptr[i+1];
        idx = proc_to_idx[proc];
        standard_start = standard_comm->send_data->indptr[idx];
        standard_end = standard_comm->send_data->indptr[idx+1];
        ASSERT_EQ(end - start, standard_end - standard_start);
        std::sort(new_comm->send_data->indices.begin() + start,
               new_comm->send_data->indices.begin() + end);
        std::sort(standard_comm->send_data->indices.begin() + standard_start,
               standard_comm->send_data->indices.begin() + standard_end);
        for (int j = 0; j < end - start; j++)
        {
            ASSERT_EQ(new_comm->send_data->indices[start+j],
                    standard_comm->send_data->indices[standard_start+j]);
        }
    }

    for (int i = 0; i < standard_comm->recv_data->num_msgs; i++)
    {
        proc = standard_comm->recv_data->procs[i];
        proc_to_idx[proc] = i;
    }
    ASSERT_EQ(standard_comm->recv_data->num_msgs, new_comm->recv_data->num_msgs);
    ASSERT_EQ(standard_comm->recv_data->size_msgs, new_comm->recv_data->size_msgs);
    for (int i = 0; i < new_comm->recv_data->num_msgs; i++)
    {
        proc = new_comm->recv_data->procs[i];
        start = new_comm->recv_data->indptr[i];
        end = new_comm->recv_data->indptr[i+1];
        idx = proc_to_idx[proc];
        standard_start = standard_comm->recv_data->indptr[idx];
        standard_end = standard_comm->recv_data->indptr[idx+1];
        ASSERT_EQ(end - start, standard_end - standard_start);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(InitCommTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double eps = 0.001;
    double theta = M_PI / 8.0;
    int grid[2] = {50, 50};
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    delete A->comm;
    A->comm = NULL;
    ParComm* standard_comm;
    ParComm* two_comm;
    ParComm* three_comm;

    int tag = 9182;
    aligned_vector<int> off_proc_col_to_proc;
    A->partition->form_col_to_proc(A->off_proc_column_map, off_proc_col_to_proc);
    A->init_tap_communicators();
    if (A->tap_comm)
    {
        A->tap_comm->n_shared--;
        A->tap_comm = NULL;
    }

    // Create normal ParComm
    A->init_par_communicator(off_proc_col_to_proc, tag);
    standard_comm = A->comm;
    A->comm = NULL;

    // Create ParComm from 2-step
    A->tap_comm = A->two_step;
    A->tap_comm->n_shared++;
    A->init_par_communicator(off_proc_col_to_proc, tag);
    two_comm = A->comm;
    A->comm = NULL;
    A->tap_comm->n_shared--;
    A->tap_comm = NULL;

    // Create ParComm from 3-step
    A->tap_comm = A->three_step;
    A->tap_comm->n_shared++;
    A->init_par_communicator(off_proc_col_to_proc, tag);
    three_comm = A->comm;
    A->comm = NULL;
    A->tap_comm->n_shared--;
    A->tap_comm = NULL;

    compare(standard_comm, two_comm);
    compare(standard_comm, three_comm);

    delete standard_comm;
    delete two_comm;
    delete three_comm;    

    delete A;

} // end of TEST(ParCommTest, TestsInCore) //
