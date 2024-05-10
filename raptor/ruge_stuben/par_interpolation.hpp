// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_DIRECT_INTERPOLATION_HPP
#define RAPTOR_PAR_DIRECT_INTERPOLATION_HPP

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"

namespace raptor {

ParCSRMatrix* direct_interpolation(ParCSRMatrix* A, 
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        bool tap_amg = false);

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        bool tap_amg = false, int num_variables = 1, int* variables = NULL);

ParCSRMatrix* extended_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const std::vector<int>& states,
        const std::vector<int>& off_proc_states,
        const double filter_threshold = 0.3,
        bool tap_amg = false, int num_variables = 1, int* variables = NULL);

ParCSRMatrix * one_point_interpolation(const ParCSRMatrix & A,
                                       const ParCSRMatrix & S,
                                       const splitting_t & splitting);
ParBSRMatrix * one_point_interpolation(const ParBSRMatrix & A,
                                       const ParCSRMatrix & S,
                                       const splitting_t & splitting);

ParBSRMatrix * local_air(const ParBSRMatrix & A,
                         const ParCSRMatrix & S,
                         const splitting_t & splitting);


} // namespace raptor

#endif
