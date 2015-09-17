#include "ef_mat.h"


PetscErrorCode ef_bmg2_SetValuesStencil(Mat mat, PetscInt m, const MatStencil idxm[], PetscInt n,
                                        const MatStencil idxn[], const PetscScalar v[],
                                        InsertMode addv)
{
	PetscInt ierr;
	ef_bmg2_mat *ctx;
	ierr = MatShellGetContext(mat, (void**) &ctx);
	unsigned int i, j;

	grid_coord *coords = (grid_coord*) malloc(n*m*sizeof(grid_coord));

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			// This logic may be incorrect
			if (idxm[i].i-1 == idxn[j].i &&
			    idxm[i].j == idxn[j].j) coords[i*n+j].dir = BMG2_W;
			if (idxm[i].i+1 == idxn[j].i &&
			    idxm[i].j == idxn[j].j) coords[i*n+j].dir = BMG2_E;
			if (idxm[i].i == idxn[j].i &&
			    idxm[i].j-1 == idxn[j].j) coords[i*n+j].dir = BMG2_S;
			if (idxm[i].i == idxn[j].i &&
			    idxm[i].j+1 == idxn[j].j) coords[i*n+j].dir = BMG2_N;
			if (idxm[i].i == idxn[j].i &&
			    idxm[i].j == idxn[j].j) coords[i*n+j].dir = BMG2_C;
			coords[i*n+j].i = idxm[i].i;
			coords[i*n+j].j = idxm[i].j;
		}
	}

	bmg2_operator_set(ctx->op, n*m, coords, (double*)v);

	free(coords);
	return 0;
}

PetscErrorCode ef_bmg2_mult(Mat A, Vec x, Vec b)
{
	PetscErrorCode ierr;
	ef_bmg2_mat *ctx;
	double *barr;
	const double *xarr;

	ierr = MatShellGetContext(A, (void**) &ctx);CHKERRQ(ierr);
	ierr = VecGetArray(b, &barr);CHKERRQ(ierr);
	ierr = VecGetArrayRead(x, &xarr);CHKERRQ(ierr);

	bmg2_operator_apply(ctx->op, xarr, barr);

	ierr = VecRestoreArrayRead(x, &xarr);CHKERRQ(ierr);
	ierr = VecRestoreArray(b, &barr);CHKERRQ(ierr);

	return 0;
}
