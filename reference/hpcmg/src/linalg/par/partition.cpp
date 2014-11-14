#include "partition.h"

namespace linalg
{
    namespace par
    {
        Partition::Partition(sys::int_t n, sys::int_t num_procs):
            num_procs(num_procs), n(n)
        {};
    }
}
