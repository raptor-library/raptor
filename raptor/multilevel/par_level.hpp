// SPDX-FileCopyrightText: RAPtor Developers
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE.txt in the top-level directory for the full license.
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
