#ifndef RAPTOR_AORTHONORMALIZATION_MGS_HPP
#define RAPTOR_AORTHONORMALIZATION_MGS_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <vector>

using namespace raptor;

void BCGS(CSRMatrix* A, BVector& Q, BVector& P);

void CGS(CSRMatrix* A, BVector& P);

#endif
