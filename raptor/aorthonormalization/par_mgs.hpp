#ifndef RAPTOR_AORTHONORMALIZATION_PAR_MGS_HPP
#define RAPTOR_AORTHONORMALIZATION_PAR_MGS_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include <vector>

using namespace raptor;

void MGS(ParCSRMatrix* A, aligned_vector<ParVector>& W, aligned_vector<aligned_vector<ParVector>>& P_list);

void MGS(ParCSRMatrix* A, aligned_vector<ParVector>& P);

#endif
