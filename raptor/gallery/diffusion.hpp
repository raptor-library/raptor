// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include <mpi.h>
#include <cmath>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

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
data_t* diffusion_stencil_2d(data_t eps = 1.0, data_t theta = 0.0);

#endif
