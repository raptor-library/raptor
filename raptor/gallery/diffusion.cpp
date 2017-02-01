// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include <mpi.h>
#include <cmath>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

#include "diffusion.hpp"
using namespace raptor;

// diffusion_stencil_2d
//
// Generate a diffusion stencil
//
// Supports isotropic diffusion (FE,FD), anisotropic diffusion (FE, FD), and
// rotated anisotropic diffusion (FD).
//
// Rotated Anisotropic diffusion in 2d of the form:
//
// -div Q A Q^T grad u
//
// Q = [cos(theta) -sin(theta)]
//     [sin(theta)  cos(theta)]
//
// A = [1          0        ]
//     [0          eps      ]
//
// Parameters
// ----------
// epsilon  : double, optional
//     Anisotropic diffusion coefficient: -div A grad u,
//     where A = [1 0; 0 epsilon].  The default is isotropic, epsilon=1.0
// theta : double, optional
//     Rotation angle `theta` in radians defines -div Q A Q^T grad,
//     where Q = [cos(`theta`) -sin(`theta`); sin(`theta`) cos(`theta`)].
// type : {'FE','FD'}
//     Specifies the discretization as Q1 finite element (FE) or 2nd order
//     finite difference (FD)
//     The default is `theta` = 0.0
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
// Notes
// -----
// Not all combinations are supported.
//
// TODO
// ----
// Add FD option
//
data_t* diffusion_stencil_2d(data_t eps, data_t theta)
{
    data_t* stencil = (data_t*) malloc (sizeof(data_t) * 9);

    data_t C = cos(theta);
    data_t S = sin(theta);
    data_t CS = C*S;
    data_t CC = C*C;
    data_t SS = S*S;

    data_t val1 =  ((-1*eps - 1)*CC + (-1*eps - 1)*SS + ( 3*eps - 3)*CS) / 6.0;
    data_t val2 =  (( 2*eps - 4)*CC + (-4*eps + 2)*SS) / 6.0;
    data_t val3 =  ((-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS) / 6.0;
    data_t val4 =  ((-4*eps + 2)*CC + ( 2*eps - 4)*SS) / 6.0;
    data_t val5 =  (( 8*eps + 8)*CC + ( 8*eps + 8)*SS) / 6.0;

    stencil[0] = val1;
    stencil[1] = val2;
    stencil[2] = val3;
    stencil[3] = val4;
    stencil[4] = val5;
    stencil[5] = val4;
    stencil[6] = val3;
    stencil[7] = val2;
    stencil[8] = val1;

    return stencil;
}

#endif
