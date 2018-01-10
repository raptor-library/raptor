#ifndef RAPTOR_KRYLOV_PAR_CG_HPP
#define RAPTOR_KRYLOV_PAR_CG_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include <vector>

using namespace raptor;

void CG(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

#endif
