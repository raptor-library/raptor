// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.

#include "laplacian27pt.hpp"

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

data_t* laplace_stencil_27pt()
{
    data_t* stencil = new data_t[27];

    for (int i = 0; i < 27; i++)
    {
        stencil[i] = -1;
    }

    stencil[13] = 26;

    return stencil;
}

}
