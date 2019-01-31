// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_MPI_TYPES_HPP_
#define RAPTOR_CORE_MPI_TYPES_HPP_

#include "types.hpp"
#include <mpi.h>

// Global Timing Variables
extern bool profile;
extern double total_t;
extern double collective_t;
extern double p2p_t;
extern double* current_t;

#define RAPtor_MPI_COMM_WORLD        MPI_COMM_WORLD
#define RAPtor_MPI_COMM_NULL         MPI_COMM_NULL

#define RAPtor_MPI_Comm              MPI_Comm
#define RAPtor_MPI_Group             MPI_Group
#define RAPtor_MPI_Datatype          MPI_Datatype
#define RAPtor_MPI_Request           MPI_Request
#define RAPtor_MPI_Status            MPI_Status
#define RAPtor_MPI_Op                MPI_Op

#define RAPtor_MPI_INT               MPI_INT
#define RAPtor_MPI_DOUBLE            MPI_DOUBLE
#define RAPtor_MPI_DOUBLE_INT        MPI_DOUBLE_INT
#define RAPtor_MPI_LONG              MPI_LONG
#define RAPtor_MPI_PACKED            MPI_PACKED

#define RAPtor_MPI_STATUS_IGNORE     MPI_STATUS_IGNORE
#define RAPtor_MPI_STATUSES_IGNORE   MPI_STATUSES_IGNORE

#define RAPtor_MPI_SOURCE            MPI_SOURCE
#define RAPtor_MPI_ANY_SOURCE        MPI_ANY_SOURCE

#define RAPtor_MPI_IN_PLACE          MPI_IN_PLACE
#define RAPtor_MPI_SUM               MPI_SUM
#define RAPtor_MPI_MAX               MPI_MAX
#define RAPtor_MPI_BOR               MPI_BOR


// MPI Information
extern int RAPtor_MPI_Comm_rank(RAPtor_MPI_Comm comm, int *rank);
extern int RAPtor_MPI_Comm_size(RAPtor_MPI_Comm comm, int *size);

// Collective Operations
extern int RAPtor_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, 
        RAPtor_MPI_Datatype datatype, RAPtor_MPI_Op op, RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, 
        RAPtor_MPI_Datatype datatype, RAPtor_MPI_Op op, int root, 
        RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Gather(const void *sendbuf, int sendcount, 
        RAPtor_MPI_Datatype sendtype, void *recvbuf, int recvcount,
        RAPtor_MPI_Datatype recvtype, int root, RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Allgather(const void* sendbuf, int sendcount,
        RAPtor_MPI_Datatype sendtype, void *recvbuf, int recvcount,
         RAPtor_MPI_Datatype recvtype, RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Allgatherv(const void* sendbuf, int sendcount,
        RAPtor_MPI_Datatype sendtype, void *recvbuf, const int *recvcounts, 
        const int* displs, RAPtor_MPI_Datatype recvtype, RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
        RAPtor_MPI_Datatype datatype, RAPtor_MPI_Op op, RAPtor_MPI_Comm comm, 
        RAPtor_MPI_Request* request);
extern int RAPtor_MPI_Ibarrier(RAPtor_MPI_Comm comm, 
        RAPtor_MPI_Request *request);
extern int RAPtor_MPI_Barrier(RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Bcast(void *buffer, int count, RAPtor_MPI_Datatype datatype,
        int root, RAPtor_MPI_Comm comm);

// Point-to-Point Operations
extern int RAPtor_MPI_Send(const void *buf, int count,
        RAPtor_MPI_Datatype datatype, int dest, int tag, RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Isend(const void *buf, int count, 
        RAPtor_MPI_Datatype datatype, int dest, int tag, RAPtor_MPI_Comm comm,
        RAPtor_MPI_Request * request);
extern int RAPtor_MPI_Issend(const void *buf, int count, 
        RAPtor_MPI_Datatype datatype, int dest, int tag, RAPtor_MPI_Comm comm,
        RAPtor_MPI_Request * request);
extern int RAPtor_MPI_Recv(void *buf, int count, RAPtor_MPI_Datatype datatype,
        int source, int tag, RAPtor_MPI_Comm comm, RAPtor_MPI_Status * status);
extern int RAPtor_MPI_Irecv(void *buf, int count, RAPtor_MPI_Datatype datatype,
        int source, int tag, RAPtor_MPI_Comm comm, RAPtor_MPI_Request * request);

// Waiting for data
extern int RAPtor_MPI_Wait(RAPtor_MPI_Request *request, 
        RAPtor_MPI_Status *status);
extern int RAPtor_MPI_Waitall(int count, RAPtor_MPI_Request array_of_requests[], 
        RAPtor_MPI_Status array_of_statuses[]);
extern int RAPtor_MPI_Probe(int source, int tag, RAPtor_MPI_Comm comm,
        RAPtor_MPI_Status* status);
extern int RAPtor_MPI_Iprobe(int source, int tag, RAPtor_MPI_Comm comm, 
        int *flag, RAPtor_MPI_Status *status);
extern int RAPtor_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
extern int RAPtor_MPI_Testall(int count, MPI_Request array_of_requests[], 
        int* flag, MPI_Status array_of_statuses[]);

// Packing Data
extern int RAPtor_MPI_Pack(const void *inbuf, int incount, 
        RAPtor_MPI_Datatype datatype, void *outbuf, int outside, int *position, 
        RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Unpack(const void *inbuf, int insize, int *position, 
        void *outbuf, int outcount, RAPtor_MPI_Datatype datatype, RAPtor_MPI_Comm comm);
extern int RAPtor_MPI_Get_count(const RAPtor_MPI_Status *status, 
        RAPtor_MPI_Datatype datatype, int *count);
extern int RAPtor_MPI_Pack_size(int incount, RAPtor_MPI_Datatype datatype, 
        RAPtor_MPI_Comm comm, int *size);

// Timing Data
extern int RAPtor_MPI_Wtime();

// Creating Communicators
extern int RAPtor_MPI_Comm_free(RAPtor_MPI_Comm *comm);
extern int RAPtor_MPI_Comm_split(RAPtor_MPI_Comm comm, int color, int key,
        RAPtor_MPI_Comm* new_comm);
extern int RAPtor_MPI_Comm_group(RAPtor_MPI_Comm comm, RAPtor_MPI_Group *group);
extern int RAPtor_MPI_Comm_create_group(RAPtor_MPI_Comm comm, RAPtor_MPI_Group group,
        int tag, RAPtor_MPI_Comm* newcomm);
extern int RAPtor_MPI_Group_incl(RAPtor_MPI_Group group, int n, const int ranks[],
        RAPtor_MPI_Group *newgroup);

#endif
