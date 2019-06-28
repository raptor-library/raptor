#ifndef RAPTOR_AORTHONORMALIZATION_PAR_CGS_TIMED_HPP
#define RAPTOR_AORTHONORMALIZATION_PAR_CGS_TIMED_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include <vector>

using namespace raptor;

void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& Q2, ParBVector& P, aligned_vector<double>& times);

void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& P, aligned_vector<double>& times);

void CGS(ParCSRMatrix* A, ParBVector& P, aligned_vector<double>& times);

#endif
