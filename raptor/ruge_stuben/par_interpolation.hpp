// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_DIRECT_INTERPOLATION_HPP
#define RAPTOR_PAR_DIRECT_INTERPOLATION_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* direct_interpolation(ParCSRMatrix* A, 
        ParCSRBoolMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states);

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRBoolMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states);

#endif

