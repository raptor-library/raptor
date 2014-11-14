#ifndef LA_PAR_PARTITION_H
#define LA_PAR_PARTITION_H

#include "sys/types.h"

namespace linalg
{
    namespace par
    {
        class Partition
        {
            public:
                Partition(sys::int_t n, sys::int_t num_procs);
                virtual ~Partition() {};
                virtual sys::int_t low(sys::int_t id) = 0;
                virtual sys::int_t high(sys::int_t id) = 0;
                virtual sys::int_t size(sys::int_t id) = 0;
                virtual sys::int_t owner(sys::int_t index) = 0;

            protected:
                sys::int_t num_procs;
                sys::int_t n;
        };
    }
}

#endif
