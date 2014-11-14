#ifndef LINALG_PAR_VECTOR_H
#define LINALG_PAR_VECTOR_H

#include <mpi.h>

#include "linalg/seq/vector.h"
#include "sys/types.h"
#include "partition.h"

namespace linalg
{
    namespace par
    {
        class Vector
        {
            public:
                Vector(MPI_Comm comm, sys::int_t size,
                       Partition* partitioning);
                Vector(MPI_Comm comm, sys::int_t size);
                ~Vector();
                sys::int_t size() const { return global_size; };
                seq::Vector &local() { return *local_vector; };
                const seq::Vector &local() const { return *local_vector; };
                void set(sys::data_t val);
                void scale(sys::data_t val);

            private:
                sys::int_t global_size;
                seq::Vector* local_vector;
                Partition* partitioning;
        };
    }
}
#endif
