// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_MIS_HPP
#define RAPTOR_AGGREGATION_MIS_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"

namespace raptor {

void mis2(CSRMatrix* A, std::vector<int>& states,
        double* rand_vals = NULL);

}
#endif
