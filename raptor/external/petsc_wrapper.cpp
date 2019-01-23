#include "petsc_wrapper.hpp"


#undef __FUNCT__
#define __FUNCT__ "raptor_pc_create"
PetscErrorCode raptor_pc_create(stella_pc **pc)
{
    raptor_shell* ctx;
    cts = (raptor_shell*) malloc (sizeof(raptor_shell));
    *pc = ctx;

	return 0;
}


#undef __FUNCT__
#define __FUNCT__ "raptor_pc_setup"
PetscErrorCode raptor_pc_setup(PC pc)
{
	#ifdef WITH_CEDAR
	PetscErrorCode ierr;
	Mat pmat;
	stella_pc *pc_ctx;
	stella_bmg_mat *mat_ctx;

	ierr = PCGetOperators(pc,PETSC_NULL,&pmat);CHKERRQ(ierr);
	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	ierr = MatShellGetContext(pmat, (void**)&mat_ctx);CHKERRQ(ierr);

	pc_ctx->nd = mat_ctx->nd;
	pc_ctx->solvec = mat_ctx->solvec;
	pc_ctx->rhsvec = mat_ctx->rhsvec;
	int cedar_err;
	cedar_err = cedar_solver_create(mat_ctx->so, &pc_ctx->solver);chkerr(cedar_err);

	if (cedar_err) {
		SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Cedar error code: %d", cedar_err);
	}
	#endif

	return 0;
}


#undef __FUNCT__
#define __FUNCT__ "raptor_pc_apply"
PetscErrorCode raptor_pc_apply(PC pc, Vec x, Vec y)
{
	#ifdef WITH_CEDAR
	int cedar_err;
	PetscErrorCode ierr;
	stella_pc *pc_ctx;
	double *yarr;
	const double *xarr;

	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	ierr = VecGetArrayRead(x, &xarr);CHKERRQ(ierr);
	ierr = VecGetArray(y, &yarr);CHKERRQ(ierr);

	cedar_copyto(xarr, pc_ctx->rhsvec);
	cedar_err = cedar_solver_run(pc_ctx->solver, pc_ctx->solvec, pc_ctx->rhsvec);chkerr(cedar_err);
	cedar_copyfrom(pc_ctx->solvec, yarr);

	ierr = VecRestoreArrayRead(x, &xarr);CHKERRQ(ierr);
	ierr = VecRestoreArray(y, &yarr);CHKERRQ(ierr);
	#endif

	return 0;
}



#undef __FUNCT__
#define __FUNCT__ "raptor_pc_destroy"
PetscErrorCode raptor_pc_destroy(PC pc)
{
	PetscErrorCode ierr;
	stella_pc *pc_ctx;

	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	free(pc_ctx);

	return 0;
}
