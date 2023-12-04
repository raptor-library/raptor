// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
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
