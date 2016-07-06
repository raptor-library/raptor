// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <mpi.h>
#include "core/types.hpp"
#include "laplacian27pt.hpp"

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
data_t* laplace_stencil_27pt()
{
    data_t* stencil = (data_t*) malloc (sizeof(data_t) * 27);

    for (int i = 0; i < 27; i++)
    {
        stencil[i] = -1;
    }

    stencil[13] = 26;

    return stencil;
}

