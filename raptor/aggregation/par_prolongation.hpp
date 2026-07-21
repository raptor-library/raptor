// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_AGGREGATION_PAR_PROLONGATION_HPP
#define RAPTOR_AGGREGATION_PAR_PROLONGATION_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/matrix_traits.hpp"
#include "raptor/core/par_vector.hpp"

namespace raptor {

enum class prolongation_weighting
{
    local,
    block
    // TODO
//     block_spectral
};

template <class T, is_bsr_or_csr<T> = true>
T* jacobi_prolongation(T* A, T* tentative, bool tap_comm = false,
        double omega = 4.0/3, int num_smooth_steps = 1, prolongation_weighting weighting =
        prolongation_weighting::local);
}
#endif
