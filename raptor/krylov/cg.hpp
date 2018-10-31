#ifndef RAPTOR_KRYLOV_CG_HPP
#define RAPTOR_KRYLOV_CG_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>

using namespace raptor;

void CG(CSRMatrix* A, Vector& x, Vector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SRE_CG(CSRMatrix* A, Vector& x, Vector& b, int t, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SRE_CG2(CSRMatrix* A, Vector& x, Vector& b, int t, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

#endif
