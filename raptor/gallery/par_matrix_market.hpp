// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
// Adapted from the NIST Matrix Market I/O library (mmio.c), which is in the
// public domain: https://math.nist.gov/MatrixMarket/mmio-c.html
/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#ifndef PAR_MM_IO_H
#define PAR_MM_IO_H

#include "matrix_market.hpp"
#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

/*  high level routines */
ParCSRMatrix* read_par_mm(const char *fname);
void write_par_mm(ParCSRMatrix* A, const char *fname);
}

#endif
