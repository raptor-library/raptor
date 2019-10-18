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

TEST(ParBVectorTinyRectangularTAPSpMVTest, TestsInUtil)
{
    setenv("PPN", "4", 1);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int vecs_in_block = 3;
    int global_rows = 16;
    int global_cols = 6;
   
    // Setup ParCSRMatrix P - 16 x 6 //
    aligned_vector<int> on_proc_idx1, on_proc_idx2;
    aligned_vector<int> off_proc_idx1, off_proc_idx2;
    aligned_vector<double> on_proc_data, off_proc_data;
    aligned_vector<int> off_col_map, on_col_map, row_map;

    if (rank == 0)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(2);
        off_proc_idx1.push_back(3);

        off_proc_idx2.push_back(0);
        off_proc_idx2.push_back(1);
        off_proc_idx2.push_back(1);

        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);

        off_col_map.push_back(4);
        off_col_map.push_back(5);

        row_map.push_back(0);
        row_map.push_back(1);
    }
    else if (rank == 1)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(1);
        on_proc_idx1.push_back(1);

        on_proc_idx2.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(1);
        off_proc_idx1.push_back(2);
        off_proc_idx1.push_back(3);

        off_proc_idx2.push_back(0);
        off_proc_idx2.push_back(1);
        off_proc_idx2.push_back(2);

        on_proc_data.push_back(1.0);

        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(2);
        off_col_map.push_back(3);
        off_col_map.push_back(5);

        on_col_map.push_back(0);
        row_map.push_back(2);
        row_map.push_back(3);
        row_map.push_back(4);
    }
    else if (rank == 2)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(1);
        on_proc_idx1.push_back(2);
        on_proc_idx1.push_back(2);

        on_proc_idx2.push_back(0);
        on_proc_idx2.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(1);
        off_proc_idx1.push_back(2);
        off_proc_idx1.push_back(3);

        off_proc_idx2.push_back(2);
        off_proc_idx2.push_back(1);
        off_proc_idx2.push_back(0);
        
        on_proc_data.push_back(1.0);
        on_proc_data.push_back(1.0);

        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(0);
        off_col_map.push_back(3);
        off_col_map.push_back(5);

        on_col_map.push_back(1);

        row_map.push_back(5);
        row_map.push_back(6);
        row_map.push_back(7);
    }
    else if (rank == 3)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(1);

        off_proc_idx2.push_back(0);
        
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(4);

        row_map.push_back(8);
    }
    else if (rank == 4)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(1);
        on_proc_idx1.push_back(1);

        on_proc_idx2.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(1);
        off_proc_idx1.push_back(2);

        off_proc_idx2.push_back(1);
        off_proc_idx2.push_back(0);
        
        on_proc_data.push_back(1.0);

        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(0);
        off_col_map.push_back(5);

        on_col_map.push_back(2);

        row_map.push_back(9);
        row_map.push_back(10);
    }
    else if (rank == 5)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(1);

        on_proc_idx2.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(1);

        off_proc_idx2.push_back(0);
        
        on_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(2);

        on_col_map.push_back(3);

        row_map.push_back(11);
    }
    else if (rank == 6)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(1);
        on_proc_idx1.push_back(2);

        on_proc_idx2.push_back(0);
        on_proc_idx2.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(1);
        off_proc_idx1.push_back(2);

        off_proc_idx2.push_back(0);
        off_proc_idx2.push_back(1);
        
        on_proc_data.push_back(1.0);
        on_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(1);
        off_col_map.push_back(2);

        on_col_map.push_back(4);

        row_map.push_back(12);
        row_map.push_back(13);
    }
    else if (rank == 7)
    {
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(0);
        on_proc_idx1.push_back(1);

        on_proc_idx2.push_back(0);

        off_proc_idx1.push_back(0);
        off_proc_idx1.push_back(2);
        off_proc_idx1.push_back(4);

        off_proc_idx2.push_back(1);
        off_proc_idx2.push_back(2);
        off_proc_idx2.push_back(0);
        off_proc_idx2.push_back(2);
        
        on_proc_data.push_back(1.0);

        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        off_proc_data.push_back(1.0);
        
        off_col_map.push_back(0);
        off_col_map.push_back(1);
        off_col_map.push_back(3);

        on_col_map.push_back(5);

        row_map.push_back(14);
        row_map.push_back(15);
    }

    int first_local_row, first_local_col;
    first_local_row = row_map[0];
    if (on_col_map.size())
    {
        first_local_col = on_col_map[0];
    }
    else
    {
        first_local_col = 0;
    }

    ParCSRMatrix* P = new ParCSRMatrix(global_rows, global_cols,
                                on_proc_idx1.size()-1, on_col_map.size(),
                                first_local_row, first_local_col);

    if (rank == 0)
    {
        P->add_value(0, 4, 1.0);
        P->add_value(0, 5, 1.0);
        P->add_value(1, 5, 1.0);
    }
    else if (rank == 1)
    {
        P->add_value(0, 5, 1.0);
        P->add_value(1, 0, 1.0);
        P->add_value(1, 2, 1.0);
        P->add_value(2, 3, 1.0);
    }
    else if (rank == 2)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(0, 5, 1.0);
        P->add_value(1, 1, 1.0);
        P->add_value(1, 3, 1.0);
        P->add_value(2, 0, 1.0);
    }
    else if (rank == 3)
    {
        P->add_value(0, 4, 1.0);
    }
    else if (rank == 4)
    {
        P->add_value(0, 2, 1.0);
        P->add_value(0, 5, 1.0);
        P->add_value(1, 0, 1.0);
    }
    else if (rank == 5)
    {
        P->add_value(0, 2, 1.0);
        P->add_value(0, 3, 1.0);
    }
    else if (rank == 6)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(0, 4, 1.0);
        P->add_value(1, 2, 1.0);
        P->add_value(1, 4, 1.0);
    }
    else if (rank == 7)
    {
        P->add_value(0, 1, 1.0);
        P->add_value(0, 3, 1.0);
        P->add_value(1, 0, 1.0);
        P->add_value(1, 3, 1.0);
        P->add_value(1, 5, 1.0);
    }
    P->on_proc->idx1.resize(on_proc_idx1.size());
    for (int i = 0; i < on_proc_idx1.size(); i++)
    {
        P->on_proc->idx1[i] = on_proc_idx1[i];
    }
    P->off_proc->idx1.resize(off_proc_idx1.size());
    for (int i = 0; i < off_proc_idx1.size(); i++)
    {
        P->off_proc->idx1[i] = off_proc_idx1[i];
    }

    P->finalize();
    
    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d on_proc ------------------\n", rank);
            P->on_proc->print();
            printf("%d off_proc ------------------\n", rank);
            P->off_proc->print();
            fflush(stdout);
            printf("%d on_proc_column_map ", rank);
            for (int i = 0; i < P->on_proc_column_map.size(); i++)
            {
                printf("%d ", P->on_proc_column_map[i]);
            }
            printf("\n%d off_proc_column_map ", rank);
            for (int i = 0; i < P->off_proc_column_map.size(); i++)
            {
                printf("%d ", P->off_proc_column_map[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Setup ParVectors //
    /*ParBVector x(global_cols, P->on_proc_num_cols, vecs_in_block);
    ParBVector b(global_rows, P->local_num_rows, vecs_in_block);*/
    
    ParVector x1(global_cols, P->on_proc_num_cols);
    ParVector x2(global_cols, P->on_proc_num_cols);
    ParVector x3(global_cols, P->on_proc_num_cols);

    ParVector b1(global_rows, P->local_num_rows);
    ParVector b2(global_rows, P->local_num_rows);
    ParVector b3(global_rows, P->local_num_rows);

    // Set values in ParVectors //
    for (int i = 0; i < x1.local_n; i++)
    {
        /*x.local->values[i] = rank;
        x.local->values[x.local_n + i] = rank*2;
        x.local->values[2*x.local_n + i] = rank*3;*/
        
        x1.local->values[i] = rank;
        x2.local->values[i] = rank*2;
        x3.local->values[i] = rank*3;
    }

    for (int p = 0; p < num_procs; p++)
    {
        if (rank == p)
        {
            printf("%d send_data ------------------\n", rank);
            for (int i = 0; i < P->comm->send_data->num_msgs; i++)
            {
                printf("%d send %d vals to %d\n", rank, 
                P->comm->send_data->indptr[i+1] - P->comm->send_data->indptr[i],
                P->comm->send_data->procs[i]); 
            }
            fflush(stdout);
            printf("%d recv_data ------------------\n", rank);
            for (int i = 0; i < P->comm->recv_data->num_msgs; i++)
            {
                printf("%d recv from %d\n", rank, 
                P->comm->recv_data->procs[i]); 
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //P->mult(x1, b1);
    
    // Setup TAPComm for tests //
    //P->tap_comm = new TAPComm(P->partition, P->off_proc_column_map);
    
    // Test tap_mult
    //P->tap_mult(x1, b1);
    //P->mult(x1, b1);
    /*P->tap_mult(x2, b2);
    P->tap_mult(x3, b3);*/

    //P->tap_mult(x, b);

    // Test each vector in block vector
    /*for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[i], b1.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[A->local_num_rows + i], b2.local->values[i], 1e-06);
    }
    for (int i = 0; i < A->local_num_rows; i++)
    {
        ASSERT_NEAR(b.local->values[2*A->local_num_rows + i], b3.local->values[i], 1e-06);
    }*/

    setenv("PPN", "16", 1);

    delete P;

} // end of TEST(ParBVectorTinyRectangularTAPSpMVTest, TestsInUtil) //
