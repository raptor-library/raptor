// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
//
#ifndef RAPTOR_GALLERY_NAP_PARTITION_HPP
#define RAPTOR_GALLERY_NAP_PARTITION_HPP

#include <mpi.h>
#include "core/types.hpp"
#include "core/mpi_types.hpp"
#include <unistd.h>
#include <set>
#include "core/par_matrix.hpp"
#include <stdio.h>
#include "metis.h"

using namespace raptor;

ParCSRMatrix* NAP_partition(ParCSRMatrix* A, aligned_vector<int>& new_rows);

#endif


