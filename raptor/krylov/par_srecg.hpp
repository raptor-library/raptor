#ifndef RAPTOR_KRYLOV_PAR_SRECG_HPP
#define RAPTOR_KRYLOV_PAR_SRECG_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/par_multilevel.hpp"
#include <vector>

using namespace raptor;

void SRECG(ParCSRMatrix* A, ParVector& x, ParVector& b, int t, aligned_vector<double>& res, 
        double tol = 1e-05, int max_iter = -1, double* comm_t = NULL);
void PSRECG(ParCSRMatrix* A, ParMultilevel* ml_single, ParMultilevel* ml, ParVector& x, ParVector& b,
        int t, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1,
        double* precond_t = NULL, double* comm_t = NULL);

#endif
