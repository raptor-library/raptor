#ifndef RAPTOR_AORTHONORMALIZATION_MGS_HPP
#define RAPTOR_AORTHONORMALIZATION_MGS_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>

using namespace raptor;

void MGS(CSRMatrix* A, aligned_vector<Vector>& W, aligned_vector<aligned_vector<Vector>>& P_list);

void MGS(CSRMatrix* A, aligned_vector<Vector>& P);

#endif
