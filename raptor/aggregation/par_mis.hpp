// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_PAR_MIS_HPP
#define RAPTOR_AGGREGATION_PAR_MIS_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

int mis2(const ParCSRMatrix* A, std::vector<int>& states, 
        std::vector<int>& off_proc_states, bool tap_comm = false, 
        double* rand_vals = NULL);

}
#endif
