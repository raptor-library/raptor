// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef LAPLACIAN27PT_HPP
#define LAPLACIAN27PT_HPP

#include <mpi.h>
#include "core/types.hpp"
#include <stdlib.h>

using namespace raptor;

/**************************************************************
 *****   27 Point Laplacian Stencil
 **************************************************************
 ***** Generate a 27-point Laplacian stencil
 *****
 ***** Returns
 ***** -------------
 ***** data_t*
 *****    A 3x3 diffusion stencil
 *****
 **************************************************************/
data_t* laplace_stencil_27pt();

#endif
