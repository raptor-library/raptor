bool profile = false;
double collective_t = 0.0;
double p2p_t = 0.0;
double* current_t;
double mat_t = 0.0;
double vec_t = 0.0;
double total_t = 0.0;

#include <mpi.h>
#include "mpi_types.hpp"

void init_profile()
{
    profile = true;
    reset_profile();
}
void reset_profile()
{
    collective_t = 0.0;
    p2p_t = 0.0;
    mat_t = 0.0;
    vec_t = 0.0;
    if (profile) total_t = -MPI_Wtime();
    else total_t = 0.0;
}
void finalize_profile()
{
    profile = false;
    total_t += MPI_Wtime();
}
void average_profile(int n_iter)
{
    total_t /= n_iter;
    collective_t /= n_iter;
    p2p_t /= n_iter;
    vec_t /= n_iter;
    mat_t /= n_iter;
}
void print_profile(const char* string)
{
    int rank;
    double t0;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    MPI_Allreduce(&total_t, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("%s Total Time: %e\n", string, t0);
    if (fabs(t0 - total_t) > zero_tol)
        reset_profile();
    MPI_Reduce(&collective_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 && t0 > 0) printf("%s Collective Comm Time: %e\n", string, t0);
    MPI_Reduce(&p2p_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 && t0 > 0) printf("%s P2P Comm Time: %e\n", string, t0);
    MPI_Reduce(&vec_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 && t0 > 0) printf("%s Vec Comm Time: %e\n", string, t0);
    MPI_Reduce(&mat_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 && t0 > 0) printf("%s Mat Comm Time: %e\n", string, t0);
}

int RAPtor_MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, 
        RAPtor_MPI_Datatype datatype, RAPtor_MPI_Op op, RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Reduce(const void *sendbuf, void *recvbuf, int count, 
        RAPtor_MPI_Datatype datatype, RAPtor_MPI_Op op, int root, RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Gather(const void *sendbuf, int sendcount, RAPtor_MPI_Datatype sendtype,
        void *recvbuf, int recvcount, RAPtor_MPI_Datatype recvtype, int root, RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, 
            recvtype, root, comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Allgather(const void* sendbuf, int sendcount, RAPtor_MPI_Datatype sendtype,
        void *recvbuf, int recvcount, RAPtor_MPI_Datatype recvtype, RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, 
            recvcount, recvtype, comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Allgatherv(const void* sendbuf, int sendcount, RAPtor_MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int* displs, 
        RAPtor_MPI_Datatype recvtype, RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
            displs, recvtype, comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
        RAPtor_MPI_Datatype datatype, RAPtor_MPI_Op op, RAPtor_MPI_Comm comm, RAPtor_MPI_Request* request)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    if (profile) current_t = &collective_t;
    return val;
}

int RAPtor_MPI_Bcast(void *buffer, int count, RAPtor_MPI_Datatype datatype,
        int root, RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Bcast(buffer, count, datatype, root, comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Ibarrier(RAPtor_MPI_Comm comm, RAPtor_MPI_Request *request)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Ibarrier(comm, request);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    if (profile) current_t = &collective_t;
    return val;
}

int RAPtor_MPI_Barrier(RAPtor_MPI_Comm comm)
{
    if (profile) collective_t -= RAPtor_MPI_Wtime();
    int val = MPI_Barrier(comm);
    if (profile) collective_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Send(const void *buf, int count, RAPtor_MPI_Datatype datatype, int dest,
        int tag, RAPtor_MPI_Comm comm)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Send(buf, count, datatype, dest, tag, comm);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Isend(const void *buf, int count, RAPtor_MPI_Datatype datatype, int dest, int tag,
        RAPtor_MPI_Comm comm, RAPtor_MPI_Request * request)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    if (profile) current_t = &p2p_t;
    return val;
}

int RAPtor_MPI_Issend(const void *buf, int count, RAPtor_MPI_Datatype datatype, int dest, int tag,
        RAPtor_MPI_Comm comm, RAPtor_MPI_Request * request)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Issend(buf, count, datatype, dest, tag, comm, request);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    if (profile) current_t = &p2p_t;
    return val;
}

int RAPtor_MPI_Recv(void *buf, int count, RAPtor_MPI_Datatype datatype, int source, int tag,
        RAPtor_MPI_Comm comm, RAPtor_MPI_Status * status)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Recv(buf, count, datatype, source, tag, comm, status);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Irecv(void *buf, int count, RAPtor_MPI_Datatype datatype, int source,
        int tag, RAPtor_MPI_Comm comm, RAPtor_MPI_Request * request)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    if (profile) current_t = &p2p_t;
    return val;
}


int RAPtor_MPI_Wait(RAPtor_MPI_Request *request, RAPtor_MPI_Status *status)
{
    if (profile) *current_t -= RAPtor_MPI_Wtime();
    int val = MPI_Wait(request, status);
    if (profile) *current_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Waitall(int count, RAPtor_MPI_Request array_of_requests[], RAPtor_MPI_Status array_of_statuses[])
{
    if (profile) *current_t -= RAPtor_MPI_Wtime();
    int val = MPI_Waitall(count, array_of_requests, array_of_statuses);
    if (profile) *current_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Probe(int source, int tag, RAPtor_MPI_Comm comm, RAPtor_MPI_Status* status)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Probe(source, tag, comm, status);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Iprobe(int source, int tag, RAPtor_MPI_Comm comm,
        int *flag, RAPtor_MPI_Status *status)
{
    if (profile) p2p_t -= RAPtor_MPI_Wtime();
    int val = MPI_Iprobe(source, tag, comm, flag, status);
    if (profile) p2p_t += RAPtor_MPI_Wtime();
    if (profile) current_t = &p2p_t;
    return val;
}

int RAPtor_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
    if (profile) *current_t -= RAPtor_MPI_Wtime();
    int val = MPI_Test(request, flag, status);
    if (profile) *current_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Testall(int count, MPI_Request array_of_requests[],
        int* flag, MPI_Status array_of_statuses[])
{
    if (profile) *current_t -= RAPtor_MPI_Wtime();
    int val = MPI_Testall(count, array_of_requests, flag, array_of_statuses);
    if (profile) *current_t += RAPtor_MPI_Wtime();
    return val;
}

int RAPtor_MPI_Pack(const void *inbuf, int incount, 
        RAPtor_MPI_Datatype datatype, void *outbuf, int outside, int *position, 
        RAPtor_MPI_Comm comm)
{
    return MPI_Pack(inbuf, incount, datatype, outbuf, outside, position, comm);
}

int RAPtor_MPI_Unpack(const void *inbuf, int insize, int *position, 
        void *outbuf, int outcount, RAPtor_MPI_Datatype datatype, RAPtor_MPI_Comm comm)
{
    return MPI_Unpack(inbuf, insize, position, outbuf, outcount, datatype, comm);
}

double RAPtor_MPI_Wtime()
{
    return MPI_Wtime();
}


int RAPtor_MPI_Get_count(const RAPtor_MPI_Status *status, 
        RAPtor_MPI_Datatype datatype, int *count)
{
    return MPI_Get_count(status, datatype, count);
}

int RAPtor_MPI_Pack_size(int incount, RAPtor_MPI_Datatype datatype, 
        RAPtor_MPI_Comm comm, int *size)
{
    return MPI_Pack_size(incount, datatype, comm, size);
}

int RAPtor_MPI_Comm_rank(RAPtor_MPI_Comm comm, int* rank)
{
    return MPI_Comm_rank(comm, rank);
}
int RAPtor_MPI_Comm_size(RAPtor_MPI_Comm comm, int* size)
{
    return MPI_Comm_size(comm, size);
}

int RAPtor_MPI_Comm_free(RAPtor_MPI_Comm *comm)
{
    return MPI_Comm_free(comm);
}

int RAPtor_MPI_Comm_split(RAPtor_MPI_Comm comm, int color, int key,
        RAPtor_MPI_Comm* new_comm)
{
    return MPI_Comm_split(comm, color, key, new_comm);
}

int RAPtor_MPI_Comm_group(RAPtor_MPI_Comm comm, RAPtor_MPI_Group *group)
{
    return MPI_Comm_group(comm, group);
}

int RAPtor_MPI_Comm_create_group(RAPtor_MPI_Comm comm, RAPtor_MPI_Group group,
        int tag, RAPtor_MPI_Comm* newcomm)
{
    return MPI_Comm_create_group(comm, group, tag, newcomm);
}

int RAPtor_MPI_Group_incl(RAPtor_MPI_Group group, int n, const int ranks[],
        RAPtor_MPI_Group *newgroup)
{
    return MPI_Group_incl(group, n, ranks, newgroup);
}

