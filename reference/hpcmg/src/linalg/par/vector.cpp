#include "vector.h"

#include "linalg/seq/vector.h"
#include "block_partition.h"

namespace linalg
{
    namespace par
    {
        Vector::Vector(MPI_Comm comm, sys::int_t size,
                       Partition* partitioning):
            global_size(size),
            partitioning(partitioning)
        {
            int rank;
            MPI_Comm_rank(comm, &rank);

            local_vector = new seq::Vector(partitioning->size(rank));
        }

        Vector::Vector(MPI_Comm comm, sys::int_t size):
            global_size(size)
        {
            int commsize, rank;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &commsize);

            partitioning = new BlockPartition(size, commsize);

            local_vector = new seq::Vector(partitioning->size(rank));
        }

        Vector::~Vector()
        {
            delete local_vector;
            delete partitioning;
        }

        void Vector::set(sys::data_t val)
        {
            local_vector->set(val);
        }

        void Vector::scale(sys::data_t val)
        {
            local_vector->scale(val);
        }

    }
}
