// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef STENCIL_HPP
#define STENCIL_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

/**************************************************************
 *****   Stencil Grid
 **************************************************************
 ***** Generate a matrix from a specified stencil, and of a
 ***** specified size;
 *****
 ***** Parameters
 ***** -------------
 ***** stencil : data_t*
 *****    Stencil of the matrix
 ***** grid : int*
 *****    Dimensions of the matrix
 ***** dim : int
 *****    Dimension of grid variable
 *****
 ***** Returns
 ***** -------------
 ***** ParMatrix*
 *****    Matrix formed from stencil
 *****
 **************************************************************/
ParMatrix* stencil_grid(data_t* stencil, int* grid, int dim);

#endif
