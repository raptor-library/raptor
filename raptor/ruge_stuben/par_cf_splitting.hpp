// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_PAR_SPLITTING_HPP
#define RAPTOR_PAR_SPLITTING_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "cf_splitting.hpp"

using namespace raptor;

void set_initial_states(ParCSRMatrix* S, aligned_vector<int>& states);
void reset_boundaries(ParCSRMatrix* S, aligned_vector<int>& states);

void cljp_main_loop(ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, double* rand_vals = NULL);
void pmis_main_loop(ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, double* rand_vals = NULL);


void split_rs(ParCSRMatrix* S, aligned_vector<int>& states, 
        aligned_vector<int>& off_proc_states, bool tap_cf = false);

void split_cljp(ParCSRMatrix* S, aligned_vector<int>& states, 
        aligned_vector<int>& off_proc_states, bool tap_cf = false, 
        double* rand_vals = NULL);

void split_falgout(ParCSRMatrix* S, aligned_vector<int>& states, 
        aligned_vector<int>& off_proc_states, bool tap_cf = false, 
        double* rand_vals = NULL);

void split_pmis(ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, bool tap_cf = false, 
        double* rand_vals = NULL);

void split_hmis(ParCSRMatrix* S, aligned_vector<int>& states,
        aligned_vector<int>& off_proc_states, bool tap_cf = false, 
        double* rand_vals = NULL);
#endif
