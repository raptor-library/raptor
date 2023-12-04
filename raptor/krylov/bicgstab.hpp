#ifndef RAPTOR_KRYLOV_BICGSTAB_HPP
#define RAPTOR_KRYLOV_BICGSTAB_HPP

#include <vector>

#include "raptor/core/types.hpp"
#include "raptor/core/matrix.hpp"
#include "raptor/core/vector.hpp"

using namespace raptor;

void BiCGStab(CSRMatrix* A, Vector& x, Vector& b, std::vector<double>& res, double tol = 1e-05, int max_iter = -1);

void test(CSRMatrix* A, Vector& x, Vector&b, std::vector<double>& res, std::vector<double>& Apr_list, 
		std::vector<double>& alpha_list, std::vector<double>& Ass_list, std::vector<double>& AsAs_list,
		std::vector<double>& omega_list, std::vector<double>& rr_list, std::vector<double>& beta_list,
		std::vector<double>& norm_list, double tol=1e-05, int max_iter = -1);

void Test_BiCGStab(CSRMatrix* A, Vector& x, Vector& b, std::vector<double>& res, std::vector<double>& Apr_list, 
		std::vector<double>& alpha_list, std::vector<double>& Ass_list, std::vector<double>& AsAs_list, 
		std::vector<double>& omega_list, std::vector<double>& rr_list, std::vector<double>& beta_list, 
		std::vector<double>& norm_list, double tol = 1e-05, int max_iter = -1);

#endif

