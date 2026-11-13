// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_AGGREGATION_PROLONGATION_HPP
#define RAPTOR_AGGREGATION_PROLONGATION_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "raptor/core/vector.hpp"

namespace raptor {
CSRMatrix* jacobi_prolongation(CSRMatrix* A, CSRMatrix* T, double omega = 4.0/3, 
        int num_smooth_steps = 1);
}
#endif
