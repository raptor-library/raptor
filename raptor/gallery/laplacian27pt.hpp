// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef LAPLACIAN27PT_HPP
#define LAPLACIAN27PT_HPP

//#include <mpi.h>
#include "raptor/core/types.hpp"
#include <stdlib.h>

namespace raptor {

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

data_t* laplace_stencil_27pt();

}
#endif
