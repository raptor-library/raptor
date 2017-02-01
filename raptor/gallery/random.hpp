// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

ParMatrix* random_mat(int global_rows, int global_cols, int nnz_per_row);

#endif
