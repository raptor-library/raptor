static char help[] = "Solves a linear system in parallel with RAPtor:\n\
    a parallel AMG solvers\n\n";        -

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
#include <petscksp.h>
#include "raptor.hpp"

/* Define context for user-provided preconditioner */
typedef struct {
  ParMultilevel* ml;
} RAPtorShellPC;

/* Declare routines for user-provided preconditioner */
extern PetscErrorCode RAPtorShellPCCreate(RAPtorShellPC**);
extern PetscErrorCode RAPtorShellPCSetUp(PC,Mat,Vec);
extern PetscErrorCode RAPtorShellPCApply(PC,Vec x,Vec y);
extern PetscErrorCode RAPtorShellPCDestroy(PC);


/***********************************************************************/
/*          Routines for a user-defined shell preconditioner           */
/***********************************************************************/

/*
   RAPtorShellPCCreate - This routine creates a user-defined
   preconditioner context.

   Output Parameter:
.  shell - user-defined preconditioner context
*/
PetscErrorCode RAPtorShellPCCreate(RAPtorShellPC **shell)
{
  RAPtorShellPC  *newctx;
  PetscErrorCode ierr;

  ierr         = PetscNew(&newctx);CHKERRQ(ierr);
  ml = new ParRugeStubenSolver();
  *shell       = newctx;
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   SampleShellPCSetUp - This routine sets up a user-defined
   preconditioner context.

   Input Parameters:
.  pc    - preconditioner object
.  pmat  - preconditioner matrix
.  x     - vector

   Output Parameter:
.  shell - fully set up user-defined preconditioner context

   Notes:
   In this example, we define the shell preconditioner to be Jacobi's
   method.  Thus, here we create a work vector for storing the reciprocal
   of the diagonal of the preconditioner matrix; this vector is then
   used within the routine SampleShellPCApply().
*/
PetscErrorCode RAPtorShellPCSetUp(PC pc,Mat pmat,Vec x)
{
  RAPtorShellPC  *shell;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
  shell->ml->setup(A);
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   SampleShellPCApply - This routine demonstrates the use of a
   user-provided preconditioner.

   Input Parameters:
+  pc - preconditioner object
-  x - input vector

   Output Parameter:
.  y - preconditioned vector

   Notes:
   This code implements the Jacobi preconditioner, merely as an
   example of working with a PCSHELL.  Note that the Jacobi method
   is already provided within PETSc.
*/
PetscErrorCode RAPtorShellPCApply(PC pc,Vec x,Vec y)
{
  SampleShellPC  *shell;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
  shell->ml->solve(x, y);

  return 0;
}
/* ------------------------------------------------------------------- */
/*
   SampleShellPCDestroy - This routine destroys a user-defined
   preconditioner context.

   Input Parameter:
.  shell - user-defined preconditioner context
*/
PetscErrorCode RAPtorShellPCDestroy(PC pc)
{
  SampleShellPC  *shell;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc,(void**)&shell);CHKERRQ(ierr);
  delete shell->ml;
  ierr = PetscFree(shell);CHKERRQ(ierr);

  return 0;
}


/*TEST

   build:
      requires: !complex !single

   test:
      nsize: 2
      args: -ksp_view -user_defined_pc -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: tsirm
      args: -m 60 -n 60 -ksp_type tsirm -pc_type ksp -ksp_monitor_short -ksp_ksp_type fgmres -ksp_ksp_rtol 1e-10 -ksp_pc_type mg -ksp_ksp_max_it 30
      timeoutfactor: 4

TEST*/
