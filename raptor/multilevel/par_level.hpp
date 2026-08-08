// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_ML_PARLEVEL_H
#define RAPTOR_ML_PARLEVEL_H

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/par_vector.hpp"
#include "raptor/core/matrix_traits.hpp"

namespace raptor
{
    template <class T, is_bsr_or_csr<T> = true>
    class ParLevel_T
    {
        public:
            ParLevel_T()
            {
                A = nullptr;
                P = nullptr;
                AP = nullptr;
                I = nullptr;
                R = nullptr;
            }

            ~ParLevel_T()
            {
                delete A;
                delete P;
                delete R;
                delete AP;
                delete I;
            }

            T* A;
            T* P;
		    T* R;
            ParVector x;
            ParVector b;
            ParVector tmp;
            std::vector<double> block_diag_inv;

            T* AP;
            T* I;
    };

    using ParLevel = ParLevel_T<ParCSRMatrix>;
}
#endif
