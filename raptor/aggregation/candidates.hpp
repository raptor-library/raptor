// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_CANDIDATES_HPP
#define RAPTOR_AGGREGATION_CANDIDATES_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"

namespace raptor {
// TODO -- currently only accepts constant vector
CSRMatrix* fit_candidates(const int n_aggs, const std::vector<int>& aggregates, 
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, double tol = 1e-10);
}
#endif
