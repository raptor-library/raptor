#include "ef_mat.h"
#include "ef_pc.h"


#undef __FUNCT__
#define __FUNCT__ "ef_pc_create"
PetscErrorCode ef_pc_create(ef_pc **pc)
{
	ef_pc *ctx;

	ctx = (ef_pc*) malloc(sizeof(ef_pc));
	*pc = ctx;

	return 0;
}


#undef __FUNCT__
#define __FUNCT__ "ef_pc_setup"
PetscErrorCode ef_pc_setup(PC pc)
{
	PetscErrorCode ierr;
	Mat pmat;
	ef_pc *pc_ctx;
	ef_bmg2_mat *mat_ctx;

	ierr = PCGetOperators(pc,PETSC_NULL,&pmat);CHKERRQ(ierr);
	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	ierr = MatShellGetContext(pmat, (void**)&mat_ctx);CHKERRQ(ierr);

	pc_ctx->solver = bmg2_solver_create(&mat_ctx->op);

	return 0;
}


#undef __FUNCT__
#define __FUNCT__ "ef_pc_apply"
PetscErrorCode ef_pc_apply(PC pc, Vec x, Vec y)
{
	PetscErrorCode ierr;
	ef_pc *pc_ctx;
	double *yarr;
	const double *xarr;

	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	ierr = VecGetArrayRead(x, &xarr);CHKERRQ(ierr);
	ierr = VecGetArray(y, &yarr);CHKERRQ(ierr);

	bmg2_solver_run(pc_ctx->solver, yarr, xarr);

	ierr = VecRestoreArrayRead(x, &xarr);CHKERRQ(ierr);
	ierr = VecRestoreArray(y, &yarr);CHKERRQ(ierr);

	return 0;
}



#undef __FUNCT__
#define __FUNCT__ "ef_pc_destroy"
PetscErrorCode ef_pc_destroy(PC pc)
{
	PetscErrorCode ierr;
	ef_pc *pc_ctx;

	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	free(pc_ctx);

	return 0;
}
