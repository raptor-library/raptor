// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "gtest/gtest.h"
#include "raptor.hpp"

using namespace raptor;

void print_mat_data(ParCSRMatrix* A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n;
    long nl;

    MPI_Reduce(&(A->local_num_rows), &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Num Rows: %d\n", n);
    MPI_Reduce(&(A->local_num_rows), &n, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Min Num Rows: %d\n", n);
    MPI_Reduce(&(A->local_num_rows), &n, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg Num Rows: %d\n", n/num_procs);

    long nnz = A->local_nnz;
    MPI_Reduce(&nnz, &nl, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max NNZ: %ld\n", nl);
    MPI_Reduce(&nnz, &nl, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Min NNZ: %ld\n", nl);
    MPI_Reduce(&nnz, &nl, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg NNZ: %ld\n", nl/num_procs);

    MPI_Reduce(&(A->comm->send_data->num_msgs), &n, 1, MPI_INT,
            MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Num Msgs: %d\n", n);
    MPI_Reduce(&(A->comm->send_data->num_msgs), &n, 1, MPI_INT,
            MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Min Num Msgs: %d\n", n);
    MPI_Reduce(&(A->comm->send_data->num_msgs), &n, 1, MPI_INT,
            MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg Num Msgs: %d\n", n/num_procs);

    MPI_Reduce(&(A->comm->send_data->size_msgs), &n, 1, MPI_INT,
            MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Max Size Msgs: %d\n", n);
    MPI_Reduce(&(A->comm->send_data->size_msgs), &n, 1, MPI_INT,
            MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Min Size Msgs: %d\n", n);
    MPI_Reduce(&(A->comm->send_data->size_msgs), &n, 1, MPI_INT,
            MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Avg Size Msgs: %d\n", n/num_procs);
    
} 


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

    double b_val;
    int grid[2] = {25, 25};
    double eps = 0.001;
    double theta = M_PI/8.0;
    double* stencil = diffusion_stencil_2d(eps, theta);
    ParCSRMatrix* A = par_stencil_grid(stencil, grid, 2);
    delete[] stencil;
    aligned_vector<int> new_rows;
    ParCSRMatrix* A_part = NAP_partition(A, new_rows);
//    int* parts = parmetis_partition(A);
//    aligned_vector<int> parmetis_rows;
//    ParCSRMatrix* A_par_metis = repartition_matrix(A, parts, parmetis_rows);

    if (rank == 0) printf("Standard A:\n");
    print_mat_data(A);

    if (rank == 0) printf("Partitioned A:\n");
    print_mat_data(A_part);

//    if (rank == 0) printf("ParMetis A:\n");
//    print_mat_data(A_par_metis);

    delete A_part;
    delete A;


} // end of TEST(ParAnisoSpMVTest, TestsInUtil) //

