// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_MULTILEVEL_SPARSIFY
#define RAPTOR_MULTILEVEL_SPARSIFY

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

void sparsify(ParCSRMatrix* A, ParCSRMatrix* P, ParCSRMatrix* I, 
        ParCSRMatrix* AP, ParCSRMatrix* Ac, const double theta = 0.1);

#endif
