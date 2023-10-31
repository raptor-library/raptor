#include <cedar/2d/mpi/grid_func.h>

#include <cedar/capi.h>

#include <cedar/interface/object.h>
#include <cedar/interface/topo.h>
#include <cedar/interface/vec.h>


cedar_vec_cont *cedar_vec_getobj(cedar_vec handle)
{
	if (CEDAR_GET_KIND(handle) != CEDAR_KIND_VEC)
		return nullptr;

	cedar_object *obj;
	int ierr = cedar_object_get(handle, &obj);
	if (ierr or (not obj))
		return nullptr;

	return reinterpret_cast<cedar_vec_cont*>(obj->ptr);
}


extern "C"
{
	int cedar_vec_create2d(cedar_topo topo, cedar_vec *vec)
	{
		using namespace cedar;
		using namespace cedar::cdr2;

		std::shared_ptr<grid_topo> topo_obj = cedar_topo_getobj(topo);
		if (not topo_obj) {
			*vec = CEDAR_VEC_NULL;
			return CEDAR_ERR_TOPO;
		}

		cedar_object *obj = cedar_object_create(CEDAR_KIND_VEC);

		cedar_vec_cont *cont = new cedar_vec_cont;
		cont->nd = 2;
		cont->topo = topo;
		cedar_object_incref(topo);
		cont->gfunc2 = std::make_unique<mpi::grid_func>(topo_obj);
		obj->ptr = reinterpret_cast<void*>(cont);
		*vec = obj->handle;

		return CEDAR_SUCCESS;
	}


	int cedar_vec_len2d(cedar_vec vec, cedar_len *ilen, cedar_len *jlen)
	{
		auto *cont = cedar_vec_getobj(vec);
		if (not cont)
			return CEDAR_ERR_VEC;

		if (cont->nd != 2)
			return CEDAR_ERR_DIM;

		auto & grid = cont->gfunc2->grid();
		*ilen = grid.nlocal(0);
		*jlen = grid.nlocal(1);

		return CEDAR_SUCCESS;
	}


	int cedar_vec_baseptr(cedar_vec vec, cedar_real **base)
	{
		auto *cont = cedar_vec_getobj(vec);
		if (not cont)
			return CEDAR_ERR_VEC;

		if (cont->nd == 2)
			*base = cont->gfunc2->data();
		else if (cont->nd == 3) {
			#ifdef ENABLE_3D
			*base = cont->gfunc3->data();
			#else
			return CEDAR_ERR_DIM;
			#endif
		} else
			return CEDAR_ERR_DIM;

		return CEDAR_SUCCESS;
	}


	int cedar_vec_getdim(cedar_vec vec, int *nd)
	{
		auto *cont = cedar_vec_getobj(vec);
		if (not cont)
			return CEDAR_ERR_VEC;

		*nd = cont->nd;

		return CEDAR_SUCCESS;
	}


	int cedar_vec_free(cedar_vec *vec)
	{
		if (CEDAR_GET_KIND(*vec) != CEDAR_KIND_TOPO)
			return CEDAR_ERR_VEC;

		cedar_object *obj;
		cedar_object_get(*vec, &obj);
		if (obj) {
			auto *cont = reinterpret_cast<cedar_vec_cont*>(obj->ptr);
			cedar_object_decref(cont->topo);
			delete cont;
			cedar_object_free(*vec);
		} else
			return CEDAR_ERR_VEC;

		*vec = CEDAR_VEC_NULL;
		return CEDAR_SUCCESS;
	}
}
