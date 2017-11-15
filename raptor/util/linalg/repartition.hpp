// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
//
#ifndef RAPTOR_GALLERY_REPARTITION_HPP
#define RAPTOR_GALLERY_REPARTITION_HPP

#include <mpi.h>
#include "core/types.hpp"
#include <unistd.h>
#include <set>
#include "core/par_matrix.hpp"
#include <stdio.h>

using namespace raptor;

ParCSRMatrix* repartition_matrix(ParCSRMatrix* A, int* partition, std::vector<int>& new_local_rows);
void make_contiguous(ParCSRMatrix* A);

#endif

