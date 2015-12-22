#include <gtest/gtest.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "../par_vector.hpp"
#include "../par_matrix.hpp"
#include "../../gallery/matrix_IO.hpp"
#include "../../gallery/stencil.hpp"
#include "../../gallery/diffusion.hpp"

#include "util/linalg/spmv.hpp"
#include "util/linalg/matmult.hpp"

TEST(core, matmult) {
	int rank;
	int comm_size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	ASSERT_EQ(comm_size, 2);

	using namespace raptor;

	data_t vA[9] = {1,1,1,1,1,1,1,1,1};
    data_t vx[9] = {2,2,2,2,2,2,2,2,2};
    
    ParMatrix* A = new ParMatrix(3,3,vA);
	ParMatrix* x = new ParMatrix(3,3,vx);
	ParMatrix* B;

	parallel_matmult(A, x, &B);

	// write comparison answer
	data_t vBtest[9] = {6,6,6,6,6,6,6,6,6};
	ParMatrix* Btest = new ParMatrix(3,3,vBtest);

    /******************************************************
    ***** Check If Matrices are Equal (by parts)
    ******************************************************/
	// Check that Diagonal Blocks are the same
    for (int i = 0; i < B->local_rows; i++)
    {
        // Are row pointers the same?
        ASSERT_EQ(B->diag->indptr[i], Btest->diag->indptr[i]); 
        for (int j = B->diag->indptr[i]; j < B->diag->indptr[i+1]; j++)
        {
            // Are columns indices the same?
            ASSERT_EQ(B->diag->indices[j], Btest->diag->indices[j]); 

            // Are values the same?
            ASSERT_EQ(B->diag->data[j], Btest->diag->data[j]); 
        }       
    }

	// Check that Off-Diagonal Blocks are the same
    for (int i = 0; i < B->offd_num_cols; i++)
    {
        // Are column pointers the same?
        ASSERT_EQ(B->offd->indptr[i], Btest->offd->indptr[i]); 
        for (int j = B->offd->indptr[i]; j < B->offd->indptr[i+1]; j++)
        {
            // Are row indices the same?
            ASSERT_EQ(B->offd->indices[j], Btest->offd->indices[j]); 

            // Are values the same?
            ASSERT_EQ(B->offd->data[j], Btest->offd->data[j]); 
        }       
    }

}
