#include "commhandle.h"

namespace linalg
{
    namespace par
    {
        CommHandle::CommHandle(CommPkg *comm_pkg,
                               sys::data_t *send_data,
                               sys::data_t *recv_data):
            comm_pkg(comm_pkg), send_data(send_data),
            recv_data(recv_data)
        {
            sys::int_t num_sends = comm_pkg->num_sends;
            sys::int_t num_recvs = comm_pkg->num_recvs;

            sys::int_t num_requests;
            sys::int_t i, j;
            int rank, num_procs;
            int ip, vec_start, vec_len;
            MPI_Comm comm = comm_pkg->comm;

            num_requests = num_sends + num_recvs;
            requests = new MPI_Request[num_requests];

            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &num_procs);

            for(i=0; i < num_recvs; i++)
            {
                ip = comm_pkg->recv_procs[i];
                vec_start = comm_pkg->recv_vec_starts[i];
                vec_len = comm_pkg->recv_vec_starts[i+1] - vec_start;
                MPI_Irecv(&recv_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, &requests[j++]);
            }

            for (i=0; i < num_sends; i++)
            {
                vec_start = comm_pkg->send_map_starts[i];
                vec_len = comm_pkg->send_map_starts[i+1] - vec_start;
                ip = comm_pkg->send_procs[i];
                MPI_Isend(&send_data[vec_start], vec_len, MPI_DOUBLE, ip,
                          0, comm, &requests[j++]);
            }
        }

        CommHandle::~CommHandle() {}
    }
}
