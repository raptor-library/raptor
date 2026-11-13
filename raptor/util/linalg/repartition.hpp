// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
//
#ifndef RAPTOR_GALLERY_REPARTITION_HPP
#define RAPTOR_GALLERY_REPARTITION_HPP

#include <mpi.h>
#include <unistd.h>
#include <set>
#include <stdio.h>

#include "raptor/core/types.hpp"
#include "raptor/core/mpi_types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, std::vector<int>& new_local_rows);
void make_contiguous(ParCSRMatrix* A);

}
#endif
