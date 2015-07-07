#ifndef RAPTOR_UTILS_LINALG_SPMV_H
#define RAPTOR_UTILS_LINALG_SPMV_H

#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
//using Eigen::VectorXd;

#include "core/Vector.hpp"
#include "core/ParMatrix.hpp"
#include "core/Matrix.hpp"
#include "core/ParVector.hpp"
using namespace raptor;

//void sequentialSPMV(CSRMatrix* A, Vector* x, Vector* y, double alpha, double beta);
//void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta);

template <typename Derived>
void sequentialSPMV(Matrix* A, const Eigen::MatrixBase<Derived> & x, Vector* y, double alpha, double beta)
{
    *y = alpha*((*(A->m))*(x)) + beta * (*y);
}

void parallelSPMV(ParMatrix* A, ParVector* x, ParVector* y, double alpha, double beta)
{
	/* y = \alpha Ax + \beta y */

	//TODO must enforce that y and x are not aliased, or this will NOT work
	//    or we could use blocking sends as long as we post the iRecvs first

	MPI_Request *send_requests, *recv_requests;
	double *recv_data;

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    ParComm* comm = A->comm;
    auto sendProcs = comm->sendProcs;
    auto recvProcs = comm->recvProcs;
    auto sendIndices = comm->sendIndices;
    auto recvIndices = comm->recvIndices;
    int tempVectorSize = comm->sumSizeRecvs;

    MPI_Request* sendRequests;
    MPI_Request* recvRequests;
    MPI_Status* sendStatus;
    MPI_Status* recvStatus;
    double* sendBuffer;
    double* recvBuffer;

    if (A->offdNumCols)
    {
	    //TODO we do not want to malloc these every time
	    sendRequests = new MPI_Request [sendProcs.size()];
	    recvRequests = new MPI_Request [recvProcs.size()];
        sendStatus = new MPI_Status [sendProcs.size()];
        recvStatus = new MPI_Status [recvProcs.size()];
        sendBuffer = new double[comm->sumSizeSends];
        recvBuffer = new double[comm->sumSizeRecvs];

	    // Begin sending and gathering off-diagonal entries
	    double* local = x->local->data();

        int begin = 0;
        int ctr = 0;
        int reqCtr = 0;
        for (auto proc : recvProcs)
        {
            int numRecv = recvIndices[proc].size();
            printf("%d recv %d from %d\n", rank, numRecv, proc);
            MPI_Irecv(&recvBuffer[begin], numRecv, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &(recvRequests[reqCtr++]));
            begin += numRecv;
        }

        begin = 0;
        reqCtr = 0;
	    for (auto proc : sendProcs)
        {
            ctr = 0;
            for (auto sendIdx : sendIndices[proc])
            {
                sendBuffer[begin + ctr] = local[sendIdx];
                ctr++;
            }
            printf("%d send %lu to %d\n", rank, sendIndices[proc].size(), proc);
            MPI_Isend(&sendBuffer[begin], ctr, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &(sendRequests[reqCtr++]));
            begin += ctr;
        }
    }

	// Add contribution of b
	y->scale(beta);

	// Compute partial SpMV with local information
	sequentialSPMV(A->diag, *(x->local), y->local, alpha, 0.0);

    // Once data is available, add contribution of off-diagonals
	// TODO Deal with new entries as they become available
	// TODO Add an error check on the status
    if (A->offdNumCols)
    {
	    MPI_Waitall(recvProcs.size(), recvRequests, recvStatus);

        //TODO -- should not need to copy the vector like this
        //Vector tmp = Vector(tempVectorSize);
        //for (int i = 0; i < tempVectorSize; i++)
        //{
        //    tmp(i) = recvBuffer[i];
        //}

        Vector tmp = Eigen::Map<Vector>(recvBuffer, tempVectorSize);
	    //Vector tmp = Vector::Map(recvBuffer, tempVectorSize);
        sequentialSPMV(A->offd, tmp, y->local, alpha, 1.0); 

	    // Be sure sends finish
	    // TODO Add an error check on the status
	    MPI_Waitall(sendProcs.size(), sendRequests, sendStatus);
//	    delete[] recv_data; 
	    delete[] sendRequests; 
	    delete[] recvRequests; 
        delete[] recvStatus;
        delete[] sendStatus;
    }
}

// void sequentialSPMV(Matrix* A, Vector* x, Vector* y, double alpha, double beta)
// {
//     *y = alpha*((*(A->m))*(*x)) + beta * (*y);
// }

#endif
