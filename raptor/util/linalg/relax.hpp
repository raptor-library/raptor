// Copyright (c) 2015-2017, RAPtor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_UTILS_LINALG_SEQ_RELAX_H
#define RAPTOR_UTILS_LINALG_SEQ_RELAX_H
#include <float.h>

#include "core/vector.hpp"
#include "core/matrix.hpp"
#include "multilevel/level.hpp"

using namespace raptor;

void jacobi(Level* l, int num_sweeps = 1, double omega = 1.0);
void jacobi(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void sor(Level* l, int num_sweeps = 1, double omega = 1.0);
void sor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);
void ssor(Level* l, int num_sweeps = 1, double omega = 1.0);
void ssor(CSRMatrix* A, Vector& b, Vector& x, Vector& tmp, 
        int num_sweeps = 1, double omega = 1.0);


#endif

