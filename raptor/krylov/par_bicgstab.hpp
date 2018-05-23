#ifndef RAPTOR_KRYLOV_PAR_BICGSTAB_HPP
#define RAPTOR_KRYLOV_PAR_BICGSTAB_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/par_multilevel.hpp"
#include <vector>

using namespace raptor;

void BiCGStab(ParCSRMatrix* A, ParVector& x, ParVector& b, 
            aligned_vector<double>& res, double tol = 1e-06, int max_iter = -1);

int PBiCGStab(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x,
            ParVector& b, aligned_vector<double>& res, double tol = 1e-06, 
            int max_iter = -1, int precond_iter = 1);

#endif
