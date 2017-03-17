// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_LEVEL_H
#define RAPTOR_CORE_LEVEL_H

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/prolongation.hpp"
#include "aggregation/candidates.hpp"

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

            CSRMatrix A;
            CSCMatrix P;
            Vector x;
            Vector b;
            Vector tmp;
    };
}
#endif
