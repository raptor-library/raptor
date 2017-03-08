// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_HPP
#define RAPTOR_AGGREGATION_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

using namespace raptor;

int standard_aggregation(CSRMatrix& S, CSCMatrix& T, std::vector<int>& c_points);

#endif
