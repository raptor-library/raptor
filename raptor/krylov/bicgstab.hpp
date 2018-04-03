#ifndef RAPTOR_KRYLOV_BICGSTAB_HPP
#define RAPTOR_KRYLOV_BICGSTAB_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>

using namespace raptor;

void BiCGStab(CSRMatrix* A, Vector& x, Vector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

#endif
