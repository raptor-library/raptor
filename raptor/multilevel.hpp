// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef MULTILEVEL_HPP
#define MULTILEVEL_HPP

#include <math.h>
#include "ParMatrix.hpp"
#include "ParVector.hpp"

class level
{
  public:
    // Constructor
    level();
    // Destructor
    ~level();

    // Data
    ParCSRMatrix* A;
    ParCSRMatrix* R;
    ParCSRMatrix* P;
};

class multilevel
{
    public:
      // Constructor
      multilevel(level* levels, void* coarse_solver);
      // Destructor
      ~multilevel();

      // print
      char* doc();
      double cycle_complexity();
      double operator_complexity();
      double grid_complexity();

      ParVector* psolve(const ParVector* b);
      void solve(const ParVector* b,
                 const ParVector* x0,
                 const double tol,
                 const int maxiter,
                 const char* cycle,
                 const char* accel,
                 double* residuals,
                 Parvector* x0);

      void* coarse_grid_solver;
};
#endif
