// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_UTIL_LINALG_LAPACK_WRAPPER_HPP
#define RAPTOR_UTIL_LINALG_LAPACK_WRAPPER_HPP

#include <vector>

namespace raptor
{

/**
 * Compute the Moore-Penrose pseudoinverse of a dense row-major matrix.
 *
 * The input and ouput martrix is row-major
 * Singular values <= rcond * largest_singular_value are truncated.  
 * negative rcond selects max(rows, cols) * machine eps.
 */
std::vector<double> pinv(const double* matrix, int rows, int cols,
        double rcond = -1.0);
} // namespace raptor

#endif
