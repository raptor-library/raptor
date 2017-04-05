// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TYPES_HPP
#define RAPTOR_CORE_TYPES_HPP

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>
#include <set>

using namespace std;

//TODO -- should be std::numeric_limits<data_t>::epsilon ...
//#define zero_tol DBL_EPSILON
#define zero_tol 1e-16
#define MPI_INDEX_T MPI_INT
#define MPI_DATA_T MPI_DOUBLE

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace raptor
{
    using data_t = double;
    using index_t = int;
    enum format_t {CSR, CSC, COO};

    struct coo_data
    {
        index_t row;
        index_t col;
        data_t value;
    };

    struct csr_data
    {
        index_t col;
        data_t value;
    };
}

#endif
