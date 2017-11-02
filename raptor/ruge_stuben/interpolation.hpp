// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_DIRECT_INTERPOLATION_HPP
#define RAPTOR_DIRECT_INTERPOLATION_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

using namespace raptor;

CSRMatrix* direct_interpolation(CSRMatrix* A, 
        CSRBoolMatrix* S, const std::vector<int>& states);

#endif

