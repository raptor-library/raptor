// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TYPES_HPP_
#define RAPTOR_CORE_TYPES_HPP_

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
    enum format_t {BSR, CSR, CSC, COO};
    enum coarsen_t {RS, CLJP, Falgout, PMIS, HMIS};
    enum interp_t {Direct, Classical, Extended};
    enum relax_t {Jacobi, SOR, SSOR};

}

#endif
