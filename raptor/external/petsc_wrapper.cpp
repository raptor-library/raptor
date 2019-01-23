#include "petsc_wrapper.hpp"


#undef __FUNCT__
#define __FUNCT__ "raptor_pc_create"
PetscErrorCode raptor_pc_create(stella_pc **pc)
{
    raptor_shell* ctx;
    ctx = (raptor_shell*) malloc (sizeof(raptor_shell));
    *pc = ctx;

	return 0;
}


#undef __FUNCT__
#define __FUNCT__ "raptor_pc_setup"
PetscErrorCode raptor_pc_setup(PC pc)
{
	PetscErrorCode ierr;
	Mat pmat;
    raptor_pc *pc_ctx;
    
	ierr = PCGetOperators(pc,PETSC_NULL,&pmat);CHKERRQ(ierr);
	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);

	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "raptor_pc_apply"
PetscErrorCode raptor_pc_apply(PC pc, Vec x, Vec y)
{
	PetscErrorCode ierr;
	raptor_shell *pc_ctx;

	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);

	return ierr;
}



#undef __FUNCT__
#define __FUNCT__ "raptor_pc_destroy"
PetscErrorCode raptor_pc_destroy(PC pc)
{
	PetscErrorCode ierr;
	raptor_shell *pc_ctx;

	ierr = PCShellGetContext(pc, (void**)&pc_ctx);CHKERRQ(ierr);
	free(pc_ctx);

	return ierr;
}
