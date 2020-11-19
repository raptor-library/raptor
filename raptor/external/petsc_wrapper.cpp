#include "petsc_wrapper.hpp"

// Convert double* from PETSc to ParVector* in RAPtor
PetscErrorCode RAPtorConvert(const double* x_ptr, ParVector* x_vec)
{
    int n = x_vec->local_n;
    std::vector<double>& x = x_vec->local.values;
    x.assign(x_ptr, x_ptr + n);

    return 0;
}

// Convert ParVector* in RAPtor to double* in PETSc
PetscErrorCode RAPtorConvert(const ParVector* x_vec, double* x_ptr)
{
    int n = x_vec->local_n;
    const std::vector<double>& x = x_vec->local.values;
    std::copy(x.begin(), x.end(), x_ptr);

    return 0;
}

PetscErrorCode RAPtorMult(Mat A, Vec x, Vec b)
{
  PetscErrorCode ierr;
  RAPtorShellMat* ctx;
  double* b_ptr;
  const double* x_ptr;

  ierr = MatShellGetContext(A, (void**) &ctx);CHKERRQ(ierr);
  ierr = VecGetArray(b, &b_ptr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &x_ptr);CHKERRQ(ierr);

  RAPtorConvert(x_ptr, ctx->rhs);
  ctx->A->mult(*(ctx->rhs), *(ctx->sol));
  RAPtorConvert(ctx->sol, b_ptr);

  ierr = VecRestoreArrayRead(x, &x_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(b, &b_ptr);CHKERRQ(ierr);

  return 0;
}


/***********************************************************************/
/*          Routines for a user-defined shell preconditioner           */
/***********************************************************************/

/*
   RAPtorShellPCCreate - This routine creates a user-defined
   preconditioner context.

   Output Parameter:
.  shell - user-defined preconditioner context
*/
PetscErrorCode RAPtorShellPCCreate(RAPtorShellPC **pc)
{
  RAPtorShellPC  *ctx;
  PetscErrorCode ierr;

  ierr         = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->ml = new ParRugeStubenSolver();
  *pc       = ctx;
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   RAPtorShellPCSetUp - This routine sets up a user-defined
   preconditioner context.

   Input Parameters:
.  pc    - preconditioner object
.  pmat  - preconditioner matrix
.  x     - vector

   Output Parameter:
.  shell - fully set up user-defined preconditioner context

   Notes:

*/
PetscErrorCode RAPtorShellPCSetUp(PC pc)
{
  RAPtorShellPC  *ctx;
  Mat pmat;
  RAPtorShellMat *mat_ctx;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);
  ierr = PCGetOperators(pc, PETSC_NULL, &pmat); CHKERRQ(ierr);
  ierr = MatShellGetContext(pmat, (void**)&mat_ctx);CHKERRQ(ierr);  
  ctx->rhs = mat_ctx->rhs;
  ctx->sol = mat_ctx->sol;
  ctx->ml->setup(mat_ctx->A);
  ctx->ml->print_hierarchy();
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   RAPtorShellPCApply - This routine demonstrates the use of a
   user-provided preconditioner.

   Input Parameters:
+  pc - preconditioner object
-  x - input vector

   Output Parameter:
.  y - preconditioned vector
*/

PetscErrorCode RAPtorShellPCApply(PC pc,Vec y,Vec x)
{
  RAPtorShellPC  *ctx;
  PetscErrorCode ierr;
  double* x_ptr;
  const double* y_ptr;
  int global_n, local_n;

  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);

  ierr = VecGetArray(x, &x_ptr);
  ierr = VecGetArrayRead(y, &y_ptr);

  RAPtorConvert(y_ptr, ctx->rhs);
  int iter = ctx->ml->solve(*(ctx->sol), *(ctx->rhs));
  ctx->ml->print_residuals(iter);
  RAPtorConvert(ctx->sol, x_ptr);

  ierr = VecRestoreArray(x, &x_ptr);
  ierr = VecRestoreArrayRead(y, &y_ptr);

  return 0;
}
/* ------------------------------------------------------------------- */
/*
   RAPtorShellPCDestroy - This routine destroys a user-defined
   preconditioner context.

   Input Parameter:
.  shell - user-defined preconditioner context
*/
PetscErrorCode RAPtorShellPCDestroy(PC pc)
{
  RAPtorShellPC  *ctx;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc,(void**)&ctx);CHKERRQ(ierr);
  delete ctx->rhs;
  delete ctx->sol;
  delete ctx->ml;
  ierr = PetscFree(ctx);CHKERRQ(ierr);

  return 0;
}


PetscErrorCode petsc_create_preconditioner(ParCSRMatrix* A, KSP* ksp_ptr, Mat* mat_ptr,
        Vec* rhs_ptr, Vec* sol_ptr)
{
    PetscErrorCode ierr = 0;

    KSP ksp;
    PC pc;
    PCType pc_type;
    Mat pmat;
    Vec rhs, sol;

    RAPtorShellPC* shell;

    // Create RAPtor Shell
    ierr = KSPCreate(RAPtor_MPI_COMM_WORLD, &ksp);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCSetType(pc, PCSHELL);CHKERRQ(ierr);
    ierr = RAPtorShellPCCreate(&shell);CHKERRQ(ierr);
    ierr = PCShellSetSetUp(pc, RAPtorShellPCSetUp);CHKERRQ(ierr);
    ierr = PCShellSetApply(pc, RAPtorShellPCApply);CHKERRQ(ierr);
    ierr = PCShellSetDestroy(pc, RAPtorShellPCDestroy);CHKERRQ(ierr);
    ierr = PCShellSetContext(pc, shell);CHKERRQ(ierr);
    ierr = PCShellSetName(pc, "RAPtor");CHKERRQ(ierr);

    // Create matrix and rhs/sol vectors
    RAPtorShellMat* mat_ctx = (RAPtorShellMat*) malloc(sizeof(RAPtorShellMat));
    mat_ctx->A = A;
    mat_ctx->rhs = new ParVector(A->global_num_rows, A->local_num_rows);
    mat_ctx->sol = new ParVector(A->global_num_rows, A->local_num_rows);

    // Initialize Matrix Shell
    ierr = MatCreateShell(RAPtor_MPI_COMM_WORLD, A->local_num_rows, A->on_proc_num_cols, 
            A->global_num_rows, A->global_num_cols, mat_ctx, &pmat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pmat, MATOP_MULT, (void(*)(void)) RAPtorMult);CHKERRQ(ierr);

    ierr = VecCreate(RAPtor_MPI_COMM_WORLD, &rhs);CHKERRQ(ierr);
    ierr = VecCreate(RAPtor_MPI_COMM_WORLD, &sol);CHKERRQ(ierr);
    ierr = VecSetType(rhs, VECMPI);CHKERRQ(ierr);
    ierr = VecSetType(sol, VECMPI);CHKERRQ(ierr);
    ierr = VecSetSizes(rhs, A->local_num_rows, A->global_num_rows);CHKERRQ(ierr);
    ierr = VecSetSizes(sol, A->local_num_rows, A->global_num_rows);CHKERRQ(ierr);
    ierr = VecSet(rhs, 1);CHKERRQ(ierr);
    ierr = VecSet(sol, 1);CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp, pmat, pmat);CHKERRQ(ierr);

    *ksp_ptr = ksp;
    *mat_ptr = pmat;
    *rhs_ptr = rhs;
    *sol_ptr = sol;

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
