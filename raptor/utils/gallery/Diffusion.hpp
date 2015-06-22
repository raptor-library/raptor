// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "ParMatrix.hpp"

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
// Add FE and FD options
//
double* diffusion_stencil_2d(double eps = 1.0, double theta = 0.0)
{
    double* stencil = (double*) malloc (sizeof(double) * 9);

    double C = cos(theta);
    double S = sin(theta);
    double CS = C*S;
    double CC = C*C;
    double SS = S*S;

    double val1 =  ((-1*eps - 1)*CC + (-1*eps - 1)*SS + ( 3*eps - 3)*CS) / 6.0;
    double val2 =  (( 2*eps - 4)*CC + (-4*eps + 2)*SS) / 6.0;
    double val3 =  ((-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS) / 6.0;
    double val4 =  ((-4*eps + 2)*CC + ( 2*eps - 4)*SS) / 6.0;
    double val5 =  (( 8*eps + 8)*CC + ( 8*eps + 8)*SS) / 6.0;

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
