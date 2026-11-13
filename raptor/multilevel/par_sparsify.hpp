// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_MULTILEVEL_SPARSIFY
#define RAPTOR_MULTILEVEL_SPARSIFY

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

void sparsify(ParCSRMatrix* A, ParCSRMatrix* P, ParCSRMatrix* I, 
        ParCSRMatrix* AP, ParCSRMatrix* Ac, const double theta = 0.1);

}
#endif
