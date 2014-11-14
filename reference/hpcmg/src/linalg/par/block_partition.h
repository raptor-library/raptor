#ifndef LINALG_PAR_BLOCKPARTITION_H
#define LINALG_PAR_BLOCKPARTITION_H

#include "sys/types.h"
#include "partition.h"

namespace linalg
{
    namespace par
    {
        class BlockPartition: public Partition
        {
            public:
                BlockPartition(sys::int_t n, sys::int_t num_procs):
                    Partition(n, num_procs) {};

                inline sys::int_t low(sys::int_t id) 
                {
                    return id * n / num_procs;
                };
                
                inline sys::int_t high(sys::int_t id)
                {
                    return low(id+1) - 1;
                };

                inline sys::int_t size(sys::int_t id)
                {
                    return high(id) - low(id) + 1;
                };

                inline sys::int_t owner(sys::int_t index)
                {
                    return (num_procs*(index+1)-1)/n;
                };
        };
    }
}

#endif
