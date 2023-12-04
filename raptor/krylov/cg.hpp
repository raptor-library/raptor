#ifndef RAPTOR_KRYLOV_CG_HPP
#define RAPTOR_KRYLOV_CG_HPP

#include <vector>

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "raptor/core/vector.hpp"

using namespace raptor;

void CG(CSRMatrix* A, Vector& x, Vector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

#endif
