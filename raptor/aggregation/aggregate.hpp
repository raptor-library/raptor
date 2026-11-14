// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_AGGREGATE_HPP
#define RAPTOR_AGGREGATION_AGGREGATE_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "mis.hpp"

namespace raptor {

int aggregate(CSRMatrix* A, CSRMatrix* S, std::vector<int>& states,
        std::vector<int>& aggregates, double* rand_vals = NULL);

}
#endif
