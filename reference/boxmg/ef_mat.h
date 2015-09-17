#ifndef EF_MAT_H
#define EF_MAT_H

#include "petscmat.h"
#include "boxmg-2d/interface/c/operator.h"

typedef struct
{
	bmg2_operator op;
} ef_bmg2_mat;


PetscErrorCode ef_bmg2_SetValuesStencil(Mat mat, PetscInt m, const MatStencil idxm[], PetscInt n,
                                        const MatStencil idxn[], const PetscScalar v[],
                                        InsertMode addv);


PetscErrorCode ef_bmg2_mult(Mat mat, Vec, Vec);

#endif
