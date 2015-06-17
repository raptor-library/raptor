#include <cassert>
#include <iostream>
#include <mpi.h>

#include <raptor/core/ParVector.hpp>


int main(int argc, char *argv[])
{
	int rank;
	int comm_size;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	assert(comm_size == 2);

	using namespace raptor;

	ParVector v(10,5);

	v.setConstValue((rank+1)*.5);

	auto norm = v.norm<2>();

	if (rank == 0) {
		std::cout << "Norm: " << norm << std::endl;
	}

	MPI_Finalize();
	return 0;
}
