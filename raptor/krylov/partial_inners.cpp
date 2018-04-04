// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "krylov/partial_inners.hpp"

using namespace raptor;

void half_inner_contig(ParVector &x, ParVector &y, half, part_global){
    /* x : ParVector for calculating inner product
     * y : ParVector for calculating inner product
     * half : which half of processes to use in inner product
     *    0 : first half
     *    1 : second half
     * part_global : number of values being used in inner product
     */

    int rank, size, comm_rank, inner_root, recv_root;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Using color to determine which processes communicate
    // for the partial inner product
    // 0: participates in the inner product
    // 1: does not participate in the inner product
    int color;
    if ((rank < size/2) && half) color = 0;
    else if ((rank < size/2) && !(half)) color = 1;
    else if ((rank >= size/2) && !(half)) color = 0;
    else color = 1;

    // Make sure that half_procs is even or adjust
    int half_procs = size/2;
    if (size % 2 != 0) half_procs++;

    // comm_rank is the "rank" within the communicator
    //comm_rank = rank % half_procs;

    // inner_root is the "root" within the comm calculating inner_prod
    // recv_root is the "root" within the commu receiving inner_prod
    if (color && half){ 
        inner_root = 0;
	recv_root = half_procs;
    }
    else{ 
	inner_root = half_procs;
	recv_root = 0;
    }

    // Communicator for Inner Product and Communicator for Receiving Half
    MPI_Comm inner_comm, recv_comm;
    if (color) MPI_Comm_split(MPI_COMM_WORLD, color, rank, &inner_comm);
    else MPI_Comm_split(MPI_COMM_WORLD, color, rank, &recv_comm);

    data_t inner_prod = 0.0;

    if (x.local_n != y.local_n){
        printf("Error. Dimensions do not match.\n");
	exit(-1);
    }

    // Perform Inner Product on Half
    if (color){
        if (x.local_n){
            inner_prod = x.local.inner_product(y.local);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &inner_prod, 1, MPI_DATA_T, MPI_SUM, inner_comm);

    // Send partial inner product from used partition to a single process
    if (color && half) MPI_Send(&inner_prod, 1, MPI_DATA_T, recv_root, 1, MPI_COMM_WORLD);
    if (!(color) && comm_rank) MPI_Recv(&inner_prod, 1, MPI_DATA_T, inner_root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Broadcast for Receiving Half
    MPI_Bcast(&inner_prod, 1, MPI_DATA_T, recv_root, recv_comm);

    // Delete communicators
    MPI_Comm_free(&inner_comm);
    MPI_Comm_free(&recv_comm);

    // Return partial inner_prod scaled by global percentage
    return ((1.0*x.global_n)/part_global) * inner_prod;
}

// ***************** THIS NEEDS TO BE UPDATED **************************
void half_inner_striped(ParVector &x, ParVector &y){
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

void half_inner_approx(ParVector &x, ParVector &y){
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
}
