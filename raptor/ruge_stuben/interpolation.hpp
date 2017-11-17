// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_DIRECT_INTERPOLATION_HPP
#define RAPTOR_DIRECT_INTERPOLATION_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"

using namespace raptor;

CSRMatrix* direct_interpolation(CSRMatrix* A, 
        CSRMatrix* S, const std::vector<int>& states);

CSRMatrix* mod_classical_interpolation(CSRMatrix* A,
        CSRMatrix* S, const std::vector<int>& states);

#endif

