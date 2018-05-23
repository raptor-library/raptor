#ifndef RAPTOR_KRYLOV_BICGSTAB_HPP
#define RAPTOR_KRYLOV_BICGSTAB_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>

using namespace raptor;

void BiCGStab(CSRMatrix* A, Vector& x, Vector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

void test(CSRMatrix* A, Vector& x, Vector&b, aligned_vector<double>& res, aligned_vector<double>& Apr_list, 
		aligned_vector<double>& alpha_list, aligned_vector<double>& Ass_list, aligned_vector<double>& AsAs_list,
		aligned_vector<double>& omega_list, aligned_vector<double>& rr_list, aligned_vector<double>& beta_list,
		aligned_vector<double>& norm_list, double tol=1e-05, int max_iter = -1);

void Test_BiCGStab(CSRMatrix* A, Vector& x, Vector& b, aligned_vector<double>& res, aligned_vector<double>& Apr_list, 
		aligned_vector<double>& alpha_list, aligned_vector<double>& Ass_list, aligned_vector<double>& AsAs_list, 
		aligned_vector<double>& omega_list, aligned_vector<double>& rr_list, aligned_vector<double>& beta_list, 
		aligned_vector<double>& norm_list, double tol = 1e-05, int max_iter = -1);

#endif

