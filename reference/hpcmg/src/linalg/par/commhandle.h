#ifndef LINALG_PAR_COMMHANDLE_H
#define LINALG_PAR_COMMHANDLE_H

#include <mpi.h>

#include "sys/types.h"

#include "comm_pkg.h"

namespace linalg
{
    namespace par
    {
        struct CommHandle
        {
            CommHandle(CommPkg *comm_pkg,
                    sys::data_t *send_data,
                    sys::data_t *recv_data);
            ~CommHandle();

            CommPkg     *comm_pkg;
            sys::data_t *send_data;
            sys::data_t *recv_data;

            sys::int_t   num_requests;
            MPI_Request *requests;
        };
    }
}
#endif
