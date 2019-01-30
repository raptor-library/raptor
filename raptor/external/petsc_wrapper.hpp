#ifndef EXTERNAL_PETSC_WRAPPER_HPP
#define EXTERNAL_PETSC_WRAPPER_HPP

static char help[] = "Solves a linear system in parallel with RAPtor:\n\
    a parallel AMG solvers\n\n"; 

/*T
   Concepts: KSP^basic parallel example
   Concepts: PC^setting a user-defined shell preconditioner
   Processors: n
T*/



/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscmat.h>
#include <petscksp.h>
#include "core/par_matrix.hpp"
#include "multilevel/par_multilevel.hpp"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"

/* Define context for user-provided preconditioner */
typedef struct {
  ParCSRMatrix* A;
  ParVector* rhs;
  ParVector* sol;
} RAPtorShellMat;

typedef struct {
    ParMultilevel* ml;
    ParVector* rhs;
    ParVector* sol;
} RAPtorShellPC;

/* Declare routines for user-provided preconditioner */
extern PetscErrorCode RAPtorShellPCCreate(RAPtorShellPC**);
extern PetscErrorCode RAPtorShellPCSetUp(PC);
extern PetscErrorCode RAPtorShellPCApply(PC,Vec x,Vec y);
extern PetscErrorCode RAPtorShellPCDestroy(PC);
extern PetscErrorCode RAPtorMult(Mat A, Vec x, Vec y);

PetscErrorCode petsc_create_preconditioner(ParCSRMatrix* A, KSP* ksp_ptr, Mat* mat_ptr,
        Vec* rhs_ptr, Vec* sol_ptr);



// TODO -- shallow copy of matrix, remove allgathers
/*static PetscErrorCode MatConvert_RAPtor(Mat A, ParCSRMatrix** A_rap_ptr)
{
  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  PetscErrorCode ierr;
  Mat_MPIAIJ     *mpimat  = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *mat;
  Mat_SeqAIJ     *on_proc = (Mat_SeqAIJ*)(mpimat->A)->data;
  Mat_SeqAIJ     *off_proc = (Mat_SeqAIJ*)(mpimat->B)->data;
  PetscInt       *on_proc_indptr = on_proc->i;
  PetscInt       *on_proc_indices = on_proc->j;
  PetscScalar    *on_proc_data = on_proc->a;
  PetscInt       *off_proc_indptr = off_proc->i;
  PetscInt       *off_proc_indices = off_proc->j;
  PetscScalar    *off_proc_data = off_proc->a;
  PetscInt       local_num_rows = A->rmap->n;
  PetscInt       local_num_cols = A->cmap->b;
  PetscInt       *global_col_map = B->garray;

  PetscInt global_num_rows, global_num_cols;
  aligned_vector<int> row_size(num_procs);
  aligned_vector<int> col_size(num_procs);
  MPI_Allgather(&local_num_rows, 1, MPI_INT, row_size.data(), 1, MPI_INT, MPI_COMM_WORLD);  
  MPI_Allgather(&local_num_cols, 1, MPI_INT, col_size.data(), 1, MPI_INT, MPI_COMM_WORLD);  
  PetscInt first_local_row = 0;
  PetscInt first_local_col = 0;
  for (int i = 0; i < rank; i++)
  {
    first_local_row += row_size[i];
    first_local_col += col_size[i];
  }
  global_num_rows = first_local_row;
  global_num_cols = first_local_col;
  for (int i = rank; i < num_procs; i++)
  {
      global_num_rows += row_size[i];
      global_num_cols += col_size[i];
  }

  ParCSRMatrix* A_rap = new ParCSRMatrix(global_num_rows, global_num_cols,
          local_num_rows, local_num_cols, first_local_row, first_local_col);
  A_rap->on_proc->nnz = on_proc->nz;
  A_rap->off_proc->nnz = off_proc->nz;
  A_rap->on_proc->idx2.resize(A_rap->on_proc->nnz);
  A_rap->on_proc->vals.resize(A_rap->on_proc->nnz);
  A_rap->off_proc->idx2.resize(A_rap->off_proc->nnz);
  A_rap->off_proc->vals.resize(A_rap->off_proc->nnz);

  for (int i = 0; i < local_num_rows; i++)
  {
      A_rap->on_proc->idx1[i+1] = on_proc_indptr[i+1];
      A_rap->off_proc->idx1[i+1] = off_proc_indptr[i+1];
  }
  for (int i = 0; i < A_rap->on_proc->nnz; i++)
  {
      A_rap->on_proc->idx2[i] = on_proc_indices[i];
      A_rap->on_proc->vals[i] = on_proc_data[i];
  }
  for (int i = 0; i < A_rap->off_proc->nnz; i++)
  {
      A_rap->off_proc->idx2[i] = global_col_map[off_proc_indices[i]];
      A_rap->off_proc->vals[i] = off_proc_data[i];
  }      

  A_rap->finalize();

  *A_rap_ptr = A_rap;
}

*/

#endif
