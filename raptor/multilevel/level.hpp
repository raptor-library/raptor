// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
#ifndef RAPTOR_ML_LEVEL_H
#define RAPTOR_ML_LEVEL_H

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "raptor/core/vector.hpp"

// Coarse Matrices (A) are CSC
// Prolongation Matrices (P) are CSC
// P^T*A*P is then CSR*(CSC*CSC) -- returns CSC Ac
namespace raptor
{
    class Level
    {
        public:

            Level()
            {
            }

            ~Level()
            {
                delete A;
                delete P;
            }

            CSRMatrix* A;
            CSRMatrix* P;
            Vector x;
            Vector b;
            Vector tmp;
    };
}
#endif
