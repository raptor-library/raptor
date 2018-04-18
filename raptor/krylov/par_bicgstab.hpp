#ifndef RAPTOR_KRYLOV_PAR_BICGSTAB_HPP
#define RAPTOR_KRYLOV_PAR_BICGSTAB_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include <vector>

using namespace raptor;

void BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SeqInner_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SeqNorm_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

void SeqInnerSeqNorm_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

void PI_BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

#endif
