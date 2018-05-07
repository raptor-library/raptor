// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_AGGREGATE_HPP
#define RAPTOR_AGGREGATION_AGGREGATE_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "aggregation/mis.hpp"

using namespace raptor;

int aggregate(CSRMatrix* A, CSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& aggregates);

#endif


