#ifndef RAPTOR_KRYLOV_PAR_CG_HPP
#define RAPTOR_KRYLOV_PAR_CG_HPP

#include <vector>

#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/par_vector.hpp"
#include "raptor/multilevel/par_multilevel.hpp"

namespace raptor {

void CG(ParCSRMatrix* A, ParVector& x, ParVector& b, std::vector<double>& res, 
        double tol = 1e-05, int max_iter = -1, double* comm_t = NULL);
void PCG(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b, 
        std::vector<double>& res, double tol = 1e-05, int max_iter = -1,
        double* precond_t = NULL, double* comm_t = NULL);

}
#endif
