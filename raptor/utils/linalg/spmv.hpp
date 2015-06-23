#include <mpi.h>
#include <Eigen/Sparse>
using Eigen::VectorXd;

#include "Vector.hpp"
#include "ParMatrix.hpp"
#include "ParVector.hpp"
using namespace raptor;

void sequentialSPMV(CSRMatrix* A, Vector* x, Vector* y, double alpha, double beta);
void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta);


void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta)
{
	/* y = \alpha Ax + \beta y */

	//TODO must enforce that y and x are not aliased, or this will NOT work
	//    or we could use blocking sends as long as we post the iRecvs first

	MPI_Request *send_requests, *recv_requests;
	double *recv_data;

	//TODO we do not want to malloc these every time
	recv_data = new double [A->globalCols];
	send_requests = new MPI_Request [send_structure.size];
	recv_requests = new MPI_Request [recv_structure.size];

	for (int i = 0; i < A->globalCols; i++)
	{
		recv_data[i] = 0;
	}

	// Begin sending and gathering off-diagonal entries
	double* local = x->getLocalVector().data();
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
	y->scale(beta);

	// Compute partial SpMV with local information
    sequentialSPMV(A->diag, x->getLocalVector(), y->getLocalVector(), alpha, 1.0); 
	
    // Once data is available, add contribution of off-diagonals
	// TODO Deal with new entries as they become available
	// TODO Add an error check on the status
	MPI_Waitall(num_recvs, recv_requests);
	Vector received_vec(recv_data, A->globalCols);
    sequentialSPMV(A->offd, received_vec, y->getLocalVector(), alpha, 1.0); 

	// Be sure sends finish
	// TODO Add an error check on the status
	MPI_Waitall(num_sends, send_requests);
	delete[] recv_data; 
	delete[] send_requests; 
	delete[] recv_requests; 
}

void sequentialSPMV(CSRMatrix* A, Vector* x, Vector* y, double alpha, double beta)
{
    y = alpha*(A->m*x) + beta * y;
}
