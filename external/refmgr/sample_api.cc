#include "sample_api.h"

REFMGR_GEN(raptor)
namespace raptor {
	struct vector {
	vector(int n) : nlocal(n) {}
		int nlocal;
	};
}

extern "C" {
	int raptor_vec_create(int n, raptor_vec *vec)
	{
		raptor_object *obj = raptor_object_create(raptor_vec_KIND);

		obj->ptr = reinterpret_cast<void*>(new raptor::vector(n));
		*vec = obj->handle;

		return RAPTOR_SUCCESS;
	}

	int raptor_vec_getsize(raptor_vec vec, int *size)
	{
		if (REFMGR_GET_KIND(vec) != raptor_vec_KIND)
			return REFMGR_ERR_KIND;

		raptor_object *obj;
		int err = raptor_object_get(vec, &obj);
		if (err or (not obj))
			return REFMGR_ERR_HANDLE;

		auto *rvec = reinterpret_cast<raptor::vector*>(obj->ptr);
		*size = rvec->nlocal;

		return RAPTOR_SUCCESS;
	}

	int raptor_vec_free(raptor_vec *vec)
	{
		if (REFMGR_GET_KIND(*vec) != raptor_vec_KIND)
			return REFMGR_ERR_KIND;

		raptor_object *obj;
		raptor_object_get(*vec, &obj);
		if (obj) {
			auto *rvec = reinterpret_cast<raptor::vector*>(obj->ptr);
			delete rvec;
			raptor_object_free(*vec);
		} else return REFMGR_ERR_HANDLE;

		*vec = raptor_vec_NULL;
		return RAPTOR_SUCCESS;
	}

	// fortran wrapper
	raptor_vec create_vec(int n)
	{
		raptor_vec vec;
		int status = raptor_vec_create(n, &vec);
		return vec;
	}

	int getsize(raptor_vec vec)
	{
		int size;
		int status = raptor_vec_getsize(vec, &size);
		return size;
	}
}
