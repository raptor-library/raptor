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

#include <cstdint>
#include <vector>
#include <stdexcept>

#define zero_tol 1e-16
#define RAPtor_MPI_INDEX_T MPI_INT
#define RAPtor_MPI_DATA_T MPI_DOUBLE

// Defines for CF splitting and aggregation
#define TmpSelection 4
#define NewSelection 3
#define NewUnselection 2
#define Selected 1
#define Unselected 0
#define Unassigned -1
#define NoNeighbors -2


// Global Timing Variables
struct PairData
{
    double val;
    int index;
};

namespace raptor
{
    using data_t = double;
    using index_t = int;
    enum strength_t {Classical, Symmetric};
    enum class strength_norm { abs };
    enum format_t {COO, CSR, CSC, BCOO, BSR, BSC};
    enum coarsen_t {RS, CLJP, Falgout, PMIS, HMIS};
    enum interp_t { Direct, ModClassical, Extended, OnePoint };
    enum restrict_t { AIR };
    enum agg_t {MIS};
    enum prolong_t {JacobiProlongation};
    enum relax_t {Jacobi, SOR, SSOR, FCJacobi};

    struct splitting_t {
	    std::vector<int> on_proc;
	    std::vector<int> off_proc;
    };
    template<typename T, typename U>
    U sum_func(const U& a, const T&b)
    {
        return a + b;
    }

    template<typename T, typename U>
    U max_func(const U& a, const T&b)
    {
        if (a > b)
        {
            return a;
        }
        else
        {
            return b;
        }
    }
}

#endif
