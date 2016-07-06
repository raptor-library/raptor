// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include <mpi.h>
#include <cmath>

#include "core/par_matrix.hpp"
#include "core/types.hpp"

using namespace raptor;

/**************************************************************
 *****   Diffusion Stencil 2D
 **************************************************************
 ***** Generate a diffusion stencil
 *****
 ***** Supports FE stencils for isotropic diffusion,
 ***** anisotropic diffusion, and rotated anisotropic diffusion.
 ***** TODO - Add FD support
 *****
 ***** Rotated anisotropic diffusion in 2d of the form:
 *****
 ***** -div Q A Q^T grad u
 *****
 ***** Q = [cos(theta) -sin(theta)]
 *****     [sin(theta)  cos(theta)]
 *****
 ***** A = [1           0         ]
 *****     [0           eps       ]
 *****
 ***** Parameters
 ***** -------------
 ***** eps : data_t (optional)
 *****    Anisotropic diffusion coefficient : -div A grad u,
 *****    where A = [1 0: 0 eps].  The default is isotropic, eps = 1.0
 ***** theta : data_t (optional)
 *****    Rotation angle 'theta' in radians defines -div Q A Q^T grad,
 *****    where Q = [cos(theta) -sin(theta); sin(theta) cos(theta)].
 *****
 ***** Returns
 ***** -------------
 ***** data_t*
 *****    A 3x3 diffusion stencil
 *****
 **************************************************************/

data_t* diffusion_stencil_2d(data_t eps = 1.0, data_t theta = 0.0);

#endif
