// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_LEVEL_H
#define RAPTOR_CORE_LEVEL_H

#include "types.hpp"
#include "par_matrix.hpp"
#include "par_vector.hpp"

/**************************************************************
 *****   Level Class
 **************************************************************
 ***** This class constructs a single level in a multigrid 
 ***** hierarchy, consisting of a coarse-grid operator,
 ***** interpolation/restriction operators, solution and right
 ***** hand side vectors (x and b).
 *****
 ***** Attributes
 ***** -------------
 ***** A : ParMatrix*
 *****    Coarse-Grid Operator
 ***** P : ParMatrix*
 *****    Prolongation Operator
 ***** x : ParVector*
 *****    Solution Vector
 ***** b : ParVector*
 *****    Right-Hand Side Vector
 ***** 
 **************************************************************/
namespace raptor
{
    class Level
    {

        public:

          /**************************************************************
          *****   Level Class Constructor
          **************************************************************
          ***** Initializes a new level
          *****
          ***** Parameters
          ***** -------------
          ***** _A : ParMatrix*
          *****    Coarse-Grid Operator
          ***** _P : ParMatrix* 
          *****    Prolongation Operator
          **************************************************************/
          Level(ParMatrix* _A, ParMatrix* _P, int _idx)
          {
              idx = _idx;
              coarsest = false;

              A = _A;
              P = _P;

              if (idx > 0)
              {
                  x = new ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
                  b = new ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
                  has_vec = true;
              }
              tmp = new ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
          }
    
          /**************************************************************
          *****   Coarse Level Class Constructor
          **************************************************************
          ***** Initializes a new coarse level
          *****
          ***** Parameters
          ***** -------------
          ***** _A : ParMatrix*
          *****    Coarse-Grid Operator
          **************************************************************/
          Level(ParMatrix* _A, int _idx)
          {
              idx = _idx;
              coarsest = true;

              A = _A;

              if (idx > 0)
              {
                  x = new ParVector(A->global_num_cols, A->local_num_cols, A->first_local_col);
                  b = new ParVector(A->global_num_rows, A->local_num_rows, A->first_local_row);
                  has_vec = true;
              }

              tmp = NULL;
              P = NULL;
          }

          Level()
          {
              P = NULL;
              tmp = NULL;
              A = NULL;
              x = NULL;
              b = NULL;
          }

        
          /**************************************************************
          *****   Level Class Destructor
          **************************************************************
          ***** Destructs level
          **************************************************************/
          ~Level()
          {
              delete A;

              if (P)
              {
                  delete P;
              }

              if (tmp)
              {
                  delete tmp;
              }

              if (idx > 0)
              {
                  delete x;
                  delete b;
              }
          }

          ParMatrix* A;
          ParMatrix* P;
          ParVector* x;
          ParVector* b;
          ParVector* tmp;
          int idx;
          bool coarsest;
          bool has_vec;

    };
}
#endif
