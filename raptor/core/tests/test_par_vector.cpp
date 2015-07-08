#include <gtest/gtest.h>

#include <mpi.h>

#include "../par_vector.hpp"

TEST(core, vecnorm) {
	int rank;
	int comm_size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	ASSERT_EQ(comm_size, 2);

	using namespace raptor;

	ParVector v(10,5);

	v.set_const_value((rank+1)*.5);

	auto norm = v.norm<2>();

	ASSERT_EQ(norm, 2.5);
}
