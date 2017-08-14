// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_PROLONGATION_HPP
#define RAPTOR_PAR_PROLONGATION_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

void cf_splitting(ParCSRMatrix* S, std::vector<int>& states);

#endif
