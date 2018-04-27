// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "gtest/gtest.h"
#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/par_matrix.hpp"
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

TEST(ParBSRMatrixTest, TestsInCore)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<int> row_ptr = {0, 3, 5, 8, 11, 13, 16};
    std::vector<int> indices = {0, 1, 4, 1, 3, 1, 2, 5, 1, 3, 4, 0, 4, 2, 4, 5};
    std::vector<double> data = {1,0,2,1, 6,7,8,2, 1,0,0,1, 1,4,5,1, 2,0,0,0, 4,3,0,0,
    				7,2,0,0, 3,0,1,0, 1,0,0,1, 1,0,2,1, 6,7,8,2, 2,0,0,0,
    				1,4,5,1, 3,0,1,0, 4,3,0,0, 7,2,0,0};
    
    std::vector<std::vector<double>> on_blocks = {{1,0,2,1}, {6,7,8,2}, {1,4,5,1}, 
		    						{4,3,0,0}, {7,2,0,0}};
    std::vector<std::vector<int>> on_indx = {{0,0}, {0,1}, {1,1}, {2,1}, {2,2}};

    std::vector<std::vector<double>> off_blocks = {{1,0,0,1}, {2,0,0,0}, {3,0,1,0}};
    std::vector<std::vector<int>> off_indx = {{0,4}, {1,3}, {2,5}};

    // Create matrices for comparison
    BSRMatrix* A_bsr = new BSRMatrix(12, 12, 2, 2, row_ptr, indices, data);
    COOMatrix* A_coo = new COOMatrix(A_bsr);
    ParBSRMatrix* A_par_bsr = new ParBSRMatrix(12, 12, 2, 2);

    // Add on_proc blocks
    for (int i=0; i<on_blocks.size(); i++){
        A_par_bsr->add_block(on_indx[i][0], on_indx[i][1], on_blocks[i]);
        A_par_bsr->add_block(on_indx[i][0]+3, on_indx[i][1]+3, on_blocks[i]);
    }

    // Add off_proc blocks
    for(int i=0; i<off_blocks.size(); i++){
        A_par_bsr->add_block(off_indx[i][0], off_indx[i][1], off_blocks[i]);
	A_par_bsr->add_block(off_indx[i][0]+3, off_indx[i][1]-3, off_blocks[i]);
    }  

    // Finalize ParBSRMatrix and create on and off process maps
    A_par_bsr->finalize(true, 2);

    // Compare nnz
    int lcl_nnz = A_par_bsr->local_nnz;
    int nnz;
    MPI_Allreduce(&lcl_nnz, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(A_bsr->nnz, nnz);

    // Compare n_blocks
    int lcl_nblocks = A_par_bsr->on_proc->idx2.size() + A_par_bsr->off_proc->idx2.size();
    int nblocks;
    MPI_Allreduce(&lcl_nblocks,& nblocks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(A_bsr->n_blocks, nblocks);

    // Create dense matrix to compare against
    std::vector<double> A_dense = A_bsr->to_dense();

    // Compare row_ptrs, indices, and data
    if (num_procs <= 1)
    {
        for (int i=0; i<A_par_bsr->on_proc->idx1.size(); i++)
	{
            ASSERT_EQ(A_bsr->idx1[i], A_par_bsr->on_proc->idx1[i]);
	}
	for (int i=0; i<A_par_bsr->on_proc->idx2.size(); i++)
	{
            ASSERT_EQ(A_bsr->idx2[i], A_par_bsr->on_proc->idx2[i]);
	}
	for (int i=0; i<A_par_bsr->on_proc->vals.size(); i++)
	{
            ASSERT_EQ(A_bsr->vals[i], A_par_bsr->on_proc->vals[i]);
	}
    
    }
    else
    {
	int block_rows = A_par_bsr->b_rows;
	int block_cols = A_par_bsr->b_cols;
	int local_rows = A_par_bsr->local_num_rows;

	for (int i = 0; i < local_rows/block_rows; i++)
	{
            int start = A_par_bsr->on_proc->idx1[i];
	    int end = A_par_bsr->on_proc->idx1[i+1];
	    for (int j = start; j < end; j++)
	    {
                int upper_i = A_par_bsr->local_row_map[i*block_rows];
		int upper_j = A_par_bsr->on_proc_column_map[(A_par_bsr->on_proc->idx2[j])*block_cols];
		int data_offset = j * block_rows * block_cols;
		for (int bi = 0; bi < block_rows; bi++)
		{
                    for (int bj = 0; bj < block_cols; bj++)
		    {
                        int glob_i = upper_i + bi;
			int glob_j = upper_j + bj;
                        int ind = bi * block_cols + bj + data_offset;
			double val = A_par_bsr->on_proc->vals[ind];
			ASSERT_NEAR(A_dense[glob_i*12+glob_j], val, zero_tol);
		    }
		}
	    }
            
	    start = A_par_bsr->off_proc->idx1[i];
	    end = A_par_bsr->off_proc->idx1[i+1];
	    for (int j = start; j < end; j++)
	    {
                int upper_i = A_par_bsr->local_row_map[i*block_rows];
		int upper_j = A_par_bsr->off_proc_column_map[(A_par_bsr->off_proc->idx2[j])*block_cols];
		int data_offset = j * block_rows * block_cols;
		for (int bi = 0; bi < block_rows; bi++)
		{
                    for (int bj = 0; bj < block_cols; bj++)
		    {
                        int glob_i = upper_i + bi;
			int glob_j = upper_j + bj;
                        int ind = bi * block_cols + bj + data_offset;
			double val = A_par_bsr->off_proc->vals[ind];
			ASSERT_NEAR(A_dense[glob_i*12+glob_j], val, zero_tol);
		    }
		}
	    }
	}
    }

    // Delete pointers
    delete A_par_bsr;
    delete A_bsr;

} // end of TEST(ParMatrixTest, TestsInCore) //
