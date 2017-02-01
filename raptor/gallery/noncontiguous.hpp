// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef NON_CONTIGUOUS_HPP
#define NON_CONTIGUOUS_HPP

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

ParMatrix* non_contiguous(int global_rows, int global_cols, 
        std::vector<coo_data>& matrix_data);

#endif
