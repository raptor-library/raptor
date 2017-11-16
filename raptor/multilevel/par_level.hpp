// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_PARLEVEL_H
#define RAPTOR_ML_PARLEVEL_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

// Coarse Matrices (A) are CSR
// Prolongation Matrices (P) are CSR
// P^T*A*P is then CSR*(CSR*CSR) -- returns CSR Ac
namespace raptor
{
    class ParLevel
    {
        public:
            ParLevel()
            {

            }

            ~ParLevel()
            {
                delete A;
                delete P;
            }

            ParCSRMatrix* A;
            ParCSRMatrix* P;
            ParVector x;
            ParVector b;
            ParVector tmp;
    };
}
#endif
