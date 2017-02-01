// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef EXXON_READER_HPP
#define EXXON_READER_HPP

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

ParMatrix* exxon_reader(char* folder, char* iname, char* fname, char* suffix, int** global_rows);

#endif
