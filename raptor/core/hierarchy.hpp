// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_HIERARCHY_H
#define RAPTOR_CORE_HIERARCHY_H

#include "types.hpp"
#include "par_matrix.hpp"
#include "par_vector.hpp"
#include "level.hpp"
#include "util/linalg/spmv.hpp"
#include "util/linalg/relax.hpp"
#include "util/linalg/gauss_elimination.hpp"

/**************************************************************
 *****   Hierarchy Class
 **************************************************************
 ***** This class constructs a multigrid hierarchy consisting
 ***** of an array of coarse-grid operators and an array of
 ***** interpolation/restriction operators.  Lists of vectors
 ***** (for x and b on each level) are also initalized.  The 
 ***** hierarchy has no more than max_num_levels levels.  If
 ***** less than that, the coarsest level size is enforced,
 ***** guaranteeing the coarsest level to have been min_coarse_size
 ***** and max_coarse_size rows.
 *****
 ***** Attributes
 ***** -------------
 ***** A_list : vector<ParMatrix*>
 *****    std::vector of Operators
 ***** P_list : vector<ParMatrix*>
 *****    std::vector of Prolongation Operators
 ***** x_list : vector<ParVector*>
 *****    std::vector of solution vectors
 ***** b_list : vector<ParVector*>
 *****    std::vector of right-hand side vectors
 ***** num_levels : index_t
 *****    Number of levels in hierarchy
 ***** max_num_levels : index_t
 *****    Maximum number of levels allowed in hierarchy
 ***** max_coarse_size : index_t
 *****    Maximum number of rows allowed on coarsest level
 ***** min_coarse_size : index_t
 *****    Minimum number of rows allowed on coarsest level
 ***** 
 ***** Methods
 ***** -------
 ***** add_level()
 *****    Insert a single value into the matrix.
 **************************************************************/
namespace raptor
{
    class Hierarchy
    {

        public:
          /**************************************************************
          *****   Matrix Class Constructor
          **************************************************************
          ***** Initializes an empty Matrix of the given format
          *****
          ***** Parameters
          ***** -------------
          ***** _nrows : index_t
          *****    Number of rows in the matrix
          ***** _ncols : index_t 
          *****    Number of columns in the matrix
          ***** _format : format_t
          *****    Format of the Matrix (COO, CSR, CSC)
          **************************************************************/
          Hierarchy(index_t _max_levels = 25, index_t _max_coarse_size = 50, index_t _min_coarse_size = 4)
          {
              max_levels = _max_levels;
              max_coarse_size = _max_coarse_size;
              min_coarse_size = _min_coarse_size;
              num_levels = 0;
          }

          ~Hierarchy()
          {

//              if (A_list[num_levels - 1] -> local_rows)
//              {
//                  delete[] A_coarse;
//                  delete[] permute_coarse;
//                  delete[] gather_sizes;
//                  delete[] gather_displs;
//              }

              for (int i = 0; i < num_levels; i++)
              {
                  delete levels[i];
              }
          }

          /***********************************************************
          *****  Add Level
          ************************************************************
          ***** Add a new level to the hierarchy
          *****
          ***** Parameters
          ***** -------------
          ***** A : ParMatrix*
          *****    Coarse-grid operator A
          ***** P : ParMatrix*
          *****    Prolongation operator P
          ************************************************************/
          void add_level(ParMatrix* A, ParMatrix* P);

          /***********************************************************
          *****  Add Level
          ************************************************************
          ***** Add coarsest level in hierarchy (no interp/restrict)
          *****
          ***** Parameters
          ***** -------------
          ***** A : ParMatrix*
          *****    Coarse-grid operator A
          ************************************************************/
          void add_level(ParMatrix* A);

          /*************************************************************
          *****   Fine Residual
          **************************************************************
          ***** Calculates r = b - Ax on fine level of hierarchy.  Also
          ***** stores residual norm and relative residual norm 
          ***** (if possible)
          *****
          ***** Returns
          ***** -------------
          ***** data_t :
          *****    Relative residual norm (or residual norm if b is zero)
          *************************************************************/
          data_t fine_residual();

          /***********************************************************
          ***** Cycle
          ************************************************************
          ***** Run a single multilevel cycle
          *****
          ***** Parameters
          ***** -------------
          ***** level : index_t
          *****    Current level of the hierarchy being solved
          ************************************************************/
          void cycle(index_t level);

          /***********************************************************
          ***** Solve
          ************************************************************
          ***** Run a single multilevel cycle
          *****
          ***** Parameters
          ***** -------------
          ***** x : ParVector* 
          *****    Fine level solution vector
          ***** b : ParVector* 
          *****    Fine level right hand side vector
          ***** relax_weight : data_t
          *****    Weight used in jacobi relaxation
          ************************************************************/
          void solve(ParVector* x, ParVector* b, data_t solve_tol = 1e-5, data_t _relax_weight = 2.0/3, int max_iterations = 100);


          index_t num_levels;
          index_t max_levels;
          index_t max_coarse_size;
          index_t min_coarse_size;
          index_t zero_b;

          data_t relax_weight;
          int presmooth_sweeps;
          int postsmooth_sweeps;
          data_t rhs_norm;

          std::vector<data_t> resid_list;
          std::vector<data_t> rel_resid_list;

          std::vector<Level*> levels;

          MPI_Comm comm_dense;

          data_t* A_coarse;
          int* permute_coarse;
          int* gather_sizes;
          int* gather_displs;
          int coarse_rows;
          int coarse_cols;
    };
}
#endif
