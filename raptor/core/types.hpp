// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TYPES_HPP
#define RAPTOR_CORE_TYPES_HPP

#include "mpi.h"
#include <float.h>
#include <math.h>

//TODO -- should be std::numeric_limits<data_t>::epsilon ...
#define zero_tol DBL_EPSILON
#define MPI_INDEX_T MPI_INT
#define MPI_DATA_T MPI_DOUBLE

namespace raptor
{
    using data_t = double;
    using index_t = int;
    enum format_t {CSR, CSC, COO};
}

#endif
