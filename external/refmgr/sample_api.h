#pragma once

#include "refmgr.h"

#define RAPTOR_SUCCESS 0


#ifdef __cplusplus
extern "C" {
#endif

REFMGR_INIT(raptor, raptor_mat, raptor_vec, raptor_config);

int raptor_vec_create(int n, raptor_vec *vec);
int raptor_vec_getsize(raptor_vec vec, int *size);
int raptor_vec_free(raptor_vec *vec);

#ifdef __cplusplus
}
#endif
