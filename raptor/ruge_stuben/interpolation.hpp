// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_DIRECT_INTERPOLATION_HPP
#define RAPTOR_DIRECT_INTERPOLATION_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"

namespace raptor {

CSRMatrix* direct_interpolation(CSRMatrix* A, 
        CSRMatrix* S, const std::vector<int>& states);

CSRMatrix* mod_classical_interpolation(CSRMatrix* A,
        CSRMatrix* S, const std::vector<int>& states, int num_variables = 1,
        int* variables = NULL);

CSRMatrix* extended_interpolation(CSRMatrix* A,
        CSRMatrix* S, const std::vector<int>& states, int num_variables = 1,
        int* variables = NULL);

}
#endif
