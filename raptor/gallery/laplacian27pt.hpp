// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef LAPLACIAN27PT_HPP
#define LAPLACIAN27PT_HPP

#include <mpi.h>
#include "core/types.hpp"
#include <stdlib.h>

using namespace raptor;

// 27 Point Laplacian Stencil
//
// Generate a 27-point laplacian stencil
//
// Returns
// -------
// stencil : numpy array
//     A 3x3 diffusion stencil
//
// See Also
// --------
// stencil_grid
//
//

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

#endif
