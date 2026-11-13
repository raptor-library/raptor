// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_UTIL_WRITER_HPP
#define RAPTOR_UTIL_WRITER_HPP

#include "raptor/core/par_matrix.hpp"

namespace raptor
{

void write(const char *fname, const ParCSRMatrix &mat);
void write(const char *fname, const ParBSRMatrix &mat);
void write(const char *fname, const BSRMatrix &mat);
void write(const char *fname, const CSRMatrix &mat);

}
#endif
