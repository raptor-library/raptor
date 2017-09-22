// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_SPLITTING_HPP
#define RAPTOR_SPLITTING_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

using namespace raptor;

void cf_splitting(CSRMatrix* S, std::vector<int>& states);

#endif
