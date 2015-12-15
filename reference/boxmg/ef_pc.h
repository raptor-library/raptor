#ifndef EF_PC_H
#define EF_PC_H

#include "petscksp.h"
#include "boxmg-2d/interface/c/solver.h"


typedef struct {
	bmg2_solver solver;
} ef_pc;


PetscErrorCode ef_pc_create(ef_pc**);

PetscErrorCode ef_pc_setup(PC);

PetscErrorCode ef_pc_apply(PC pc, Vec x, Vec y);

PetscErrorCode ef_pc_destroy(PC);

#endif
