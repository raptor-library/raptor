// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef LINEAR_ELASTICITY_HPP
#define LINEAR_ELASTICITY_HPP

#include <mpi.h>
#include <float.h>
#include <cmath>
#include <stdlib.h>
#include <Eigen/Dense>
#include "core/par_matrix.hpp"
#include "core/types.hpp"
#include "util/linalg/matmult.hpp"

Eigen::MatrixXd* q12d_local(data_t* vertices, data_t lame, data_t mu);

ParMatrix* linear_elasticity(index_t* grid, ParMatrix** B, data_t E = 1.0e5, data_t nu = 0.3, index_t dirichlet = 1, data_t* spacing = NULL);

#endif
