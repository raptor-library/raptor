#ifndef EXTERNAL_PETSC_WRAPPER_HPP
#define EXTERNAL_PETSC_WRAPPER_HPP

#include "petscksp.h"
#include "ruge_stuben/par_ruge_stuben_solver.hpp"
#include "aggregation/par_smoothed_aggregation_solver.hpp"

typedef struct {
    int nd;
    ParMultilevel* ml;
} raptor_shell;

PetscErrorCode raptor_pc_create(stella_pc**);

PetscErrorCode raptor_pc_setup(PC);

PetscErrorCode raptor_pc_apply(PC pc, Vec x, Vec y);

PetscErrorCode raptor_pc_destroy(PC);

#endif
