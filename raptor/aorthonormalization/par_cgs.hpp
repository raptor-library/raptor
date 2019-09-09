#ifndef RAPTOR_AORTHONORMALIZATION_PAR_MGS_HPP
#define RAPTOR_AORTHONORMALIZATION_PAR_MGS_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

using namespace raptor;

void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& Q2, ParBVector& P);

void BCGS(ParCSRMatrix* A, ParBVector& Q1, ParBVector& P);

void CGS(ParCSRMatrix* A, ParBVector& P);

void MGS(ParCSRMatrix* A, ParBVector& P);

#endif
