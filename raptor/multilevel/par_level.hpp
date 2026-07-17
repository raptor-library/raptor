// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_PARLEVEL_H
#define RAPTOR_ML_PARLEVEL_H

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/par_vector.hpp"

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
                A = NULL;
                P = NULL;
                AP = NULL;
                I = NULL;
            }

            ~ParLevel()
            {
                delete A;
                delete P;

                delete AP;
                delete I;
            }

            ParCSRMatrix* A;
            ParCSRMatrix* P;
		    ParCSRMatrix* R;
            ParVector x;
            ParVector b;
            ParVector tmp;

            ParCSRMatrix* AP;
            ParCSRMatrix* I;
    };
}
#endif
