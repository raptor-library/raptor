#include <mpi.h>
#include <Eigen/Sparse>
using Eigen::VectorXd;

#include "ParMatrix.hpp"
#include "ParVector.hpp"

void ParMatrix::spmv(ParVector *x, ParVector *b, double alpha, double beta)
{
	/* b = \alpha Ax + \beta b */

	//TODO must enforce that b and x are not aliased, or this will NOT work
	//    or we could use blocking sends as long as we post the iRecvs first

	MPI_Request *send_requests, *recv_requests;
	double *recv_data;

	//TODO we do not want to malloc these every time
	recv_data = malloc(sizeof(double)*this.globalCols);
	send_requests = malloc(sizeof(MPI_Request)*send_structure.size);
	recv_requests = malloc(sizeof(MPI_Request)*recv_structure.size);

	for (int i = 0; i < this.globalCols; i++)
	{
		recv_data[i] = 0;
	}

	// Begin sending and gathering off-diagonal entries
	double* local = x.getLocalVector()->data();
	for (send_thing in send_structure)
	{
		MPI_Isend(&(local[send_thing]), size_of_send, MPI_DOUBLE,
				  receiving_proc, 0, MPI_COMM_WORLD,
				  &(send_requests[index_of_request]));
	}
	for (recv_thing in recv_structure)
	{
		MPI_Irecv(&(recv_data[recv_thing]), size_of_recv, MPI_DOUBLE,
				  sending_proc, 0, MPI_COMM_WORLD,
				  &(recv_requests[index_of_request]));
	}

	// Add contribution of b
	b->scale(beta);

	// Compute partial SpMV with local information
	b->axpy(*(this.diag) * *(x->getLocalVector()), 1.0);

	// Once data is available, add contribution of off-diagonals
	// TODO Deal with new entries as they become available
	// TODO Add an error check on the status
	MPI_Waitall(num_recvs, recv_requests);
	Eigen::Map<VectorXd> received_vec(recv_data);
	b->axpy(*(this.offd) * received_vec, 1.0);

	// Be sure sends finish
	// TODO Add an error check on the status
	MPI_Waitall(num_sends, send_requests);
}
