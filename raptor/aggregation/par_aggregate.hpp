// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_PAR_AGGREGATE_HPP
#define RAPTOR_AGGREGATION_PAR_AGGREGATE_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "par_mis.hpp"

namespace raptor {

int aggregate(ParCSRMatrix* A, ParCSRMatrix* S, std::vector<int>& states,
        std::vector<int>& off_proc_states, std::vector<int>& aggregates,
        bool tap_comm = false, double* rand_vals = NULL);

}
#endif



