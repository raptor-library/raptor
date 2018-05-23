// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_DIRECT_INTERPOLATION_HPP
#define RAPTOR_PAR_DIRECT_INTERPOLATION_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"

using namespace raptor;

ParCSRMatrix* direct_interpolation(ParCSRMatrix* A, 
        ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        data_t* comm_t = NULL);

ParCSRMatrix* mod_classical_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        bool tap_amg = false, int num_variables = 1, int* variables = NULL,
        data_t* comm_t = NULL, data_t* comm_mat_t = NULL);

ParCSRMatrix* extended_interpolation(ParCSRMatrix* A,
        ParCSRMatrix* S, const aligned_vector<int>& states,
        const aligned_vector<int>& off_proc_states,
        bool tap_amg = false, int num_variables = 1, int* variables = NULL,
        data_t* comm_t = NULL, data_t* comm_mat_t = NULL);

#endif

