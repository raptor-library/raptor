#ifndef LINALG_PAR_COMM_PKG_H
#define LINALG_PAR_COMM_PKG_H

#include <mpi.h>

#include "sys/types.h"
#include "csrmatrix.h"

namespace linalg
{
    namespace par
    {
        class CSRMatrix;
        class CommPkg
        {
            public:
                CommPkg(CSRMatrix*);
                MPI_Comm comm;

                sys::int_t  num_sends;
                sys::int_t *send_procs;
                sys::int_t *send_map_starts;
                sys::int_t *send_map_elmts;

                sys::int_t  num_recvs;
                sys::int_t *recv_procs;
                sys::int_t *recv_vec_starts;

                MPI_Datatype *send_mpi_types;
                MPI_Datatype *recv_mpi_types;
        };
    }
}

#endif
