// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_PAR_CANDIDATES_HPP
#define RAPTOR_AGGREGATION_PAR_CANDIDATES_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/matrix_traits.hpp"

namespace raptor {

template <class T, is_bsr_or_csr<T> = true>
T* fit_candidates(T* A, const int n_aggs,
        const std::vector<int>& aggregates, 
        const std::vector<double>& B, std::vector<double>& R,
        int num_candidates, bool tap_comm = false, double tol = 1e-10);
}
#endif
