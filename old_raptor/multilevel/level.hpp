// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_LEVEL_H
#define RAPTOR_ML_LEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"

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
