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

	ASSERT_EQ(comm_size, 1);

	using namespace raptor;

	data_t vA[9] = {1,0,0,0,0,0,0,0,0};
    data_t vx[15] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    ParMatrix* A = new ParMatrix(3,3,vA);
	ParMatrix* x = new ParMatrix(3,5,vx);
	ParMatrix* B;

	parallel_matmult(A, x, &B);

	// write comparison answer
	data_t vBtest[15] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	ParMatrix* Btest = new ParMatrix(3,5,vBtest);

	ASSERT_EQ(B, Btest);
}
