#ifndef RAPTOR_CAPI_H
#define RAPTOR_CAPI_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAPTOR_SUCCESS     0
#define RAPTOR_ERR_KIND    2
#define RAPTOR_ERR_MAT     4
#define RAPTOR_ERR_VEC     5
#define RAPTOR_ERR_SOLVER  8
#define RAPTOR_ERR_CONFIG  9
#define RAPTOR_ERR_FNAME   10
#define RAPTOR_ERR_CONFIG_PATH 11
#define RAPTOR_ERR_OTHER   12

typedef int raptor_mat;
typedef int raptor_vec;
typedef int raptor_solver;

#define RAPTOR_TOPO_NULL ((raptor_topo) 0x10000000)
#define RAPTOR_MAT_NULL ((raptor_mat) 0x30000000)
#define RAPTOR_SOLVER_NULL ((raptor_solver) 0x20000000)
#define RAPTOR_VEC_NULL ((raptor_vec) 0x40000000)
#define RAPTOR_CONFIG_NULL ((raptor_config) 0x50000000)

int cedar_vec_create(cedar_vec *vec);
// int cedar_vec_baseptr(cedar_vec vec, cedar_real **base);
// int cedar_vec_len2d(cedar_vec vec, cedar_len *ilen, cedar_len *jlen);
// int cedar_vec_getdim(cedar_vec vec, int *nd);
int cedar_vec_free(cedar_vec *vec);

// int cedar_matvec(cedar_mat mat, cedar_vec x, cedar_vec y);
// int cedar_solver_create(cedar_mat mat, cedar_solver *solver);
// int cedar_solver_run(cedar_solver solver, cedar_vec x, cedar_vec b);
int cedar_solver_destroy(cedar_solver *solver);

#ifdef __cplusplus
}
#endif

#endif
