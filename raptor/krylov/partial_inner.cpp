// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/partial_inner.hpp"

using namespace raptor;

data_t half_inner_contig(ParVector &x, ParVector &y, int half, int part_global){
    /* x : ParVector for calculating inner product
     * y : ParVector for calculating inner product
     * half : which half of processes to use in inner product
     *    0 : first half
     *    1 : second half
     * part_global : number of values being used in inner product
     */

    int rank, num_procs, comm_rank, inner_root, recv_root, color;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    data_t inner_prod = 0.0;

    // Check if single process
    if (num_procs <= 1){
	if (x.local_n != y.local_n){
            printf("Error. Dimensions do not match.\n");
	    exit(-1);
	}
	inner_prod = x.local.inner_product(y.local);
	return inner_prod;
    }

    // Make sure that half_procs is even or adjust
    int half_procs = num_procs/2;
    if (num_procs % 2 != 0) half_procs++;

    // Using color to determine which processes communicate
    // for the partial inner product
    // 0: participates in the inner product
    // 1: does not participate in the inner product
    if (half){
        if (rank >= half_procs) color = 0;
	else color = 1;
	inner_root = half_procs;
	recv_root = 0;
    }
    else{	
	if (rank < half_procs) color = 0;
	else color = 1;
        inner_root = 0;
	recv_root = half_procs;
    }

    // Communicator for Inner Product and Communicator for Receiving Half
    MPI_Comm inner_comm, recv_comm;
    int inner_comm_size, recv_comm_size;
    if (!(color)){
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
        MPI_Comm_size(inner_comm, &inner_comm_size);
    }
    else{
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &recv_comm);
        MPI_Comm_size(recv_comm, &recv_comm_size);
    }

    if (x.local_n != y.local_n){
        printf("Error. Dimensions do not match.\n");
	exit(-1);
    }

    // Perform Inner Product on Half
    if (!(color)){
        if (x.local_n){
            inner_prod = x.local.inner_product(y.local);
        }
        if (inner_comm_size > 1) MPI_Allreduce(MPI_IN_PLACE, &inner_prod, 1, MPI_DATA_T, MPI_SUM, inner_comm);
    }

    //printf("rank %d inner_prod %lg\n", rank, inner_prod);
    // Send partial inner product from used partition to a single process
    if (rank == inner_root) MPI_Send(&inner_prod, 1, MPI_DATA_T, recv_root, 1, MPI_COMM_WORLD);
    if (rank == recv_root) MPI_Recv(&inner_prod, 1, MPI_DATA_T, inner_root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Broadcast for Receiving Half
    if (color && recv_comm_size > 1){
        MPI_Bcast(&inner_prod, 1, MPI_DATA_T, 0, recv_comm);
    }

    // Delete communicators
    if (!(color)) MPI_Comm_free(&inner_comm);
    else MPI_Comm_free(&recv_comm);

    // Return partial inner_prod scaled by global percentage
    return ((1.0*x.global_n)/part_global) * inner_prod;
}

// Sequential Inner Product for Testing Reproducibility
data_t sequential_inner(ParVector &x, ParVector &y){
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    data_t inner_prod = 0.0;

    // Check if single process
    if (num_procs <= 1){
	if (x.local_n != y.local_n){
            printf("Error. Dimensions do not match.\n");
	    exit(-1);
	}
	inner_prod = x.local.inner_product(y.local);
	return inner_prod;
    }

    if (rank > 1)
    {
        MPI_Recv(&inner_prod, 1, MPI_DATA_T, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for(int i=0; i<x.local_n; i++){
        inner_prod += x.local[i] * y.local[i];
    }

    if (rank < num_procs-1)
    {
        MPI_Send(&inner_prod, 1, MPI_DATA_T, rank+1, 1, MPI_COMM_WORLD);
    }

    MPI_Bcast(&inner_prod, 1, MPI_DATA_T, num_procs-1, MPI_COMM_WORLD);

    return inner_prod;
}

data_t sequential_norm(ParVector &x, index_t p){
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    data_t norm;

    norm = sequential_inner(x, x);

    return pow(norm, 1./p);
}







// ***************** THIS NEEDS TO BE UPDATED **************************
data_t half_inner_striped(ParVector &x, ParVector &y, int half, int part_global){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // CREATE COMMUNICATOR FOR EVEN/ODD RANKS

    data_t inner_prod = 0.0;

    if(x.local_n != y.local_n){
        printf("Error. Dimensions do not match.\n");
	exit(-1);
    }

    if(x.local_n){
        inner_prod = x.local.inner_product(y.local);
    }

    // CHANGE THIS TO BE CORRECT COMMUNICATOR
    MPI_Allreduce(MPI_IN_PLACE, &inner_prod, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);

    return 2 * inner_prod;
}

/*void half_inner_approx(ParVector &x, ParVector &y){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // CREATE COMMUNICATOR FOR CONTIGUOUS HALF
    // CALCULATE PERCENTAGE OF VECTOR USED

    data_t inner_prod = 0.0;

    if(x.local_n != y.local_n){
        printf("Error. Dimensions do not match.\n");
	exit(-1);
    }

    if (x.local_n){
        inner_prod = x.local.inner_product(y.local);
    }

    // CHANGE THIS TO BE CORRECT COMMUNICATOR
    MPI_Allreduce(MPI_IN_PLACE, &inner_prod, 1, MPI_DATA_T, MPI_SUM, MPI_COMM_WORLD);

    //inner_prod *= 
    return inner_prod;
}*/
